#!/usr/bin/env python3
"""
Smart Stage Brain — Dual-Layer Classroom Orchestrator
Combines calendar schedule + V-JEPA video classification to drive
ambient music, live captions, and a real-time web dashboard.

Architecture:
  Pi 5 (this script) = brain + sensors
  GPU (100.113.55.109:8766) = V-JEPA classify server

Threads:
  main        — asyncio event loop, WebSocket :8091, HTTP :8090
  camera x3   — OAK-D-POE capture, YOLO person detection, frame buffer
  vjepa       — every 10s: best camera buffer → mp4 → POST GPU → classify
  brain       — every 3s: calendar × V-JEPA → state machine → actions
  audio       — arecord → sherpa-onnx ASR (gated by asr_active)
"""

import argparse
import asyncio
import functools
import http.server
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

log = logging.getLogger("smart-stage")

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

CAMERAS = [
    {"ip": "169.254.1.10", "name": "Cam 1", "id": "cam1"},
    {"ip": "169.254.1.11", "name": "Cam 2", "id": "cam2"},
    {"ip": "169.254.1.222", "name": "Cam 3", "id": "cam3"},
]

CLASSIFY_URL = "http://100.113.55.109:8766/classify"
SCHEDULE_FILE = Path.home() / "smart-stage" / "schedule.json"
AUDIO_DIR = Path.home() / "smart-stage" / "audio"
STAGE_DIR = Path.home() / "smart-stage"

CAM_FPS = 5
CAM_W, CAM_H = 512, 288
VJEPA_INTERVAL = 10          # seconds between V-JEPA classifications
VJEPA_CLIP_SECS = 3
VJEPA_CLIP_FPS = 5
BRAIN_INTERVAL = 3           # seconds between state machine ticks
MODE_DEBOUNCE = 15           # seconds before committing mode switch

HTTP_PORT = 8090
WS_PORT = 8091

# ASR config
SAMPLE_RATE = 16000
BLOCK_MS = 200
ALSA_DEVICE = "hw:2,0"

TRIGGER_WORDS = {"HEY SMART OBJECT", "SMART OBJECT", "HEY SMART OBJECTS"}
_va_active = False

MODEL_DIR = Path.home() / "smart-stage" / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

# Env
def _load_env():
    env = {}
    for p in [Path("/etc/smart-stage.env"), STAGE_DIR / ".env"]:
        try:
            for line in open(p):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env.setdefault(k.strip(), v.strip())
        except FileNotFoundError:
            pass
    return env

_ENV = _load_env()
DISCORD_WEBHOOK = _ENV.get("DISCORD_WEBHOOK_URL", "")
DEEPSEEK_KEY = _ENV.get("DEEPSEEK_API_KEY", "")

# Voice-assistant credentials (read from env; see .env.example)
GEMINI_API_KEY = _ENV.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
FG_TTS_URL = _ENV.get("FIGURATE_TTS_URL", "https://disciplined-amazement-production.up.railway.app/api/voice/tts")
FG_API_KEY = _ENV.get("FIGURATE_API_KEY", "")
FG_CHAR_ID = _ENV.get("FIGURATE_CHAR_ID", "")

# ═══════════════════════════════════════════════════════════════════════════
# Shared state (protected by lock)
# ═══════════════════════════════════════════════════════════════════════════

lock = threading.Lock()
shutdown_event = threading.Event()

camera_counts: dict[str, int] = {}
camera_frames: dict[str, bytes] = {}       # cam_id -> JPEG bytes
camera_status: dict[str, str] = {}
frame_buffers: dict[str, list] = {}         # cam_id -> [(ts, np.ndarray)]
FRAME_BUF_MAX = VJEPA_CLIP_FPS * (VJEPA_CLIP_SECS + 1)

vjepa_result: dict = {"class": "unknown", "confidence": 0, "probabilities": {}, "ts": 0}

calendar_state: str = "no_class"
current_class: dict | None = None
next_class: dict | None = None

stage_state: str = "idle"
stage_submode: str = ""
stage_state_since: float = time.time()
prev_music_cmd: str = "none"

asr_active: bool = False
caption_text: str = ""
caption_partial: str = ""
caption_segments: list[str] = []

ws_clients: set = set()
ws_loop = None

# ═══════════════════════════════════════════════════════════════════════════
# Calendar
# ═══════════════════════════════════════════════════════════════════════════

def load_schedule():
    try:
        with open(SCHEDULE_FILE) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Schedule load failed: %s", e)
        return []

def compute_calendar_state(schedule, now=None):
    """Returns (state, current_class, next_class)."""
    if now is None:
        now = datetime.now()
    for cls in schedule:
        start = datetime.fromisoformat(cls["start_time"])
        end = datetime.fromisoformat(cls["end_time"])
        if start - timedelta(minutes=15) <= now < start:
            return "pre_class", cls, None
        if start <= now < end - timedelta(minutes=5):
            return "in_class", cls, None
        if end - timedelta(minutes=5) <= now <= end:
            return "class_ending", cls, None
        if end < now < end + timedelta(minutes=10):
            return "post_class", cls, None
    upcoming = [c for c in schedule if datetime.fromisoformat(c["start_time"]) > now]
    nxt = min(upcoming, key=lambda c: datetime.fromisoformat(c["start_time"])) if upcoming else None
    return "no_class", None, nxt

# ═══════════════════════════════════════════════════════════════════════════
# Camera threads
# ═══════════════════════════════════════════════════════════════════════════

def camera_thread(cam_info):
    import depthai as dai
    name, cam_id, ip = cam_info["name"], cam_info["id"], cam_info["ip"]

    while not shutdown_event.is_set():
        device = None
        try:
            dev_info = dai.DeviceInfo(ip)
            device = dai.Device(dev_info)
            pipeline = dai.Pipeline(device)

            model_desc = dai.NNModelDescription("luxonis/yolov6-nano:r2-coco-512x288", platform="RVC2")
            nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc))

            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(CAM_W, CAM_H)
            cam.setInterleaved(False)
            cam.setFps(CAM_FPS)
            cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

            nn = pipeline.create(dai.node.DetectionNetwork)
            nn.setConfidenceThreshold(0.5)
            nn.input.setBlocking(False)
            nn.setNNArchive(nn_archive)
            cam.preview.link(nn.input)

            det_q = nn.out.createOutputQueue(maxSize=1, blocking=False)
            rgb_q = nn.passthrough.createOutputQueue(maxSize=1, blocking=False)

            pipeline.start()
            log.info("[%s] Online (%s)", name, ip)
            with lock:
                camera_status[cam_id] = "online"

            while pipeline.isRunning() and not shutdown_event.is_set():
                det_data = det_q.tryGet()
                rgb_data = rgb_q.tryGet()

                if rgb_data is not None:
                    frame = rgb_data.getCvFrame()
                    # Draw detections
                    if det_data is not None:
                        for d in det_data.detections:
                            if d.label == 0 and d.confidence >= 0.5:
                                h, w = frame.shape[:2]
                                cv2.rectangle(frame,
                                    (int(d.xmin * w), int(d.ymin * h)),
                                    (int(d.xmax * w), int(d.ymax * h)),
                                    (0, 255, 0), 2)
                    # Store JPEG for dashboard
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with lock:
                        camera_frames[cam_id] = jpeg.tobytes()
                        # Frame buffer for V-JEPA
                        buf = frame_buffers.setdefault(cam_id, [])
                        buf.append((time.time(), rgb_data.getCvFrame()))
                        if len(buf) > FRAME_BUF_MAX:
                            buf.pop(0)

                if det_data is not None:
                    persons = [d for d in det_data.detections if d.label == 0 and d.confidence >= 0.5]
                    with lock:
                        camera_counts[cam_id] = len(persons)

                if det_data is None and rgb_data is None:
                    time.sleep(0.02)

            pipeline.stop()

        except Exception as e:
            log.warning("[%s] Error: %s. Retry in 10s...", name, e)
            with lock:
                camera_counts[cam_id] = 0
                camera_status[cam_id] = "offline"
        finally:
            if device:
                try: device.close()
                except: pass
        if not shutdown_event.is_set():
            shutdown_event.wait(10)

# ═══════════════════════════════════════════════════════════════════════════
# V-JEPA thread
# ═══════════════════════════════════════════════════════════════════════════

def vjepa_thread():
    global vjepa_result
    needed = VJEPA_CLIP_FPS * VJEPA_CLIP_SECS

    while not shutdown_event.is_set():
        shutdown_event.wait(VJEPA_INTERVAL)
        if shutdown_event.is_set():
            break

        # Pick camera with most people (or first available)
        with lock:
            best_cam = max(camera_counts, key=camera_counts.get) if camera_counts else None
            if not best_cam:
                best_cam = next(iter(frame_buffers), None)
            frames = list(frame_buffers.get(best_cam, [])) if best_cam else []

        if len(frames) < needed:
            log.debug("V-JEPA: not enough frames (%d/%d)", len(frames), needed)
            continue

        clip_frames = frames[-needed:]

        # Encode to mp4
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            h, w = clip_frames[0][1].shape[:2]
            writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), VJEPA_CLIP_FPS, (w, h))
            for _, frame in clip_frames:
                writer.write(frame)
            writer.release()

            with open(tmp_path, 'rb') as f:
                resp = requests.post(
                    CLASSIFY_URL,
                    files={"video": ("clip.mp4", f, "video/mp4")},
                    data={"camera_id": best_cam or "unknown"},
                    timeout=20,
                )
            if resp.status_code == 200:
                result = resp.json()
                with lock:
                    vjepa_result = {
                        "class": result.get("class", "unknown"),
                        "confidence": result.get("confidence", 0),
                        "probabilities": result.get("probabilities", {}),
                        "ts": time.time(),
                        "latency_ms": result.get("latency_ms", 0),
                        "camera": best_cam,
                    }
                log.info("V-JEPA: %s (%.1f%%) via %s [%dms]",
                    result.get("class"), result.get("confidence", 0) * 100,
                    best_cam, result.get("latency_ms", 0))
            else:
                log.warning("V-JEPA: HTTP %d", resp.status_code)
        except requests.Timeout:
            log.warning("V-JEPA: timeout")
        except Exception as e:
            log.warning("V-JEPA: %s", e)
        finally:
            try: os.unlink(tmp_path)
            except: pass

# ═══════════════════════════════════════════════════════════════════════════
# Brain thread (state machine)
# ═══════════════════════════════════════════════════════════════════════════

_candidate_state = None
_candidate_since = 0.0

def brain_thread():
    global stage_state, stage_submode, stage_state_since, asr_active
    global calendar_state, current_class, next_class
    global _candidate_state, _candidate_since, prev_music_cmd

    schedule = load_schedule()
    last_schedule_load = time.time()

    while not shutdown_event.is_set():
        shutdown_event.wait(BRAIN_INTERVAL)
        if shutdown_event.is_set():
            break

        # Reload schedule every 60s
        if time.time() - last_schedule_load > 60:
            schedule = load_schedule()
            last_schedule_load = time.time()

        # Calendar layer
        now = datetime.now()
        cal_state, cal_class, cal_next = compute_calendar_state(schedule, now)
        with lock:
            calendar_state = cal_state
            current_class = cal_class
            next_class = cal_next

        # V-JEPA layer
        with lock:
            vj = dict(vjepa_result)
            total_people = sum(camera_counts.values()) if camera_counts else 0

        vjepa_class = vj.get("class", "unknown")
        vjepa_age = time.time() - vj.get("ts", 0)

        # Determine target state
        new_state = stage_state
        new_submode = stage_submode
        music_cmd = "none"

        if cal_state == "no_class":
            if vjepa_class == "empty_room" or total_people == 0:
                new_state = "idle"
                new_submode = ""
                music_cmd = "stop"
            else:
                # People present but no class — just monitor
                new_state = "idle"
                new_submode = ""

        elif cal_state == "pre_class":
            if total_people > 0 or vjepa_class != "empty_room":
                new_state = "pre_class"
                new_submode = ""
                music_cmd = "welcome"
            else:
                new_state = "idle"
                new_submode = ""

        elif cal_state == "in_class":
            new_state = "class_active"
            if vjepa_class == "lecture":
                new_submode = "lecture"
                music_cmd = "stop"
            elif vjepa_class == "group_work":
                new_submode = "group_work"
                music_cmd = "bg_light"
            elif vjepa_class == "individual_work":
                new_submode = "individual_work"
                music_cmd = "stop"
            elif vjepa_class == "empty_room":
                # Disagree: calendar says class but room looks empty → restrain
                new_submode = stage_submode or "lecture"
                music_cmd = prev_music_cmd

        elif cal_state == "class_ending":
            if total_people > 0:
                new_state = "class_ending"
                new_submode = ""
                music_cmd = "farewell"
            else:
                new_state = "post_class"
                new_submode = ""
                music_cmd = "stop"

        elif cal_state == "post_class":
            new_state = "post_class"
            new_submode = ""
            music_cmd = "stop"

        # Apply debounce for state changes
        target = f"{new_state}/{new_submode}"
        current = f"{stage_state}/{stage_submode}"

        if target != current:
            if _candidate_state != target:
                _candidate_state = target
                _candidate_since = time.time()
            elif time.time() - _candidate_since >= MODE_DEBOUNCE:
                # Commit state change
                old_state = stage_state
                with lock:
                    stage_state = new_state
                    stage_submode = new_submode
                    stage_state_since = time.time()
                _candidate_state = None
                log.info("STATE: %s/%s → %s/%s", old_state, stage_submode, new_state, new_submode)

                # ASR control
                asr_active = (new_state == "class_active" and new_submode == "lecture")
        else:
            _candidate_state = None

        # Music command (only send on change)
        if music_cmd != prev_music_cmd:
            prev_music_cmd = music_cmd
            _send_music_command(music_cmd)

        # Write status
        _write_status()

_music_proc: subprocess.Popen | None = None

# YouTube URLs for each mood (lofi/chill streams and tracks)
MUSIC_URLS = {
    "welcome":  "https://www.youtube.com/watch?v=jfKfPfyJRdk",   # Lofi Girl - chill beats
    "bg_light": "https://www.youtube.com/watch?v=rUxyKA_-grg",   # Lofi Girl - jazz/chill
    "farewell": "https://www.youtube.com/watch?v=1fueZCTYkpA",   # Relaxing piano
    "chime":    None,  # Use local wav for chime
}

def _send_music_command(cmd):
    global _music_proc
    # Stop any current playback
    if _music_proc is not None:
        try:
            os.killpg(os.getpgid(_music_proc.pid), signal.SIGTERM)
            _music_proc.wait(timeout=3)
        except:
            try: os.killpg(os.getpgid(_music_proc.pid), signal.SIGKILL)
            except: pass
        _music_proc = None

    if cmd == "stop":
        log.info("MUSIC: stop")
    elif cmd == "chime":
        # Local chime sound
        audio_file = AUDIO_DIR / "chime.wav"
        if audio_file.exists():
            subprocess.Popen(["aplay", "-D", "plughw:2,0", str(audio_file)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info("MUSIC: chime")
    elif cmd in MUSIC_URLS and MUSIC_URLS[cmd]:
        url = MUSIC_URLS[cmd]
        local_wav = AUDIO_DIR / f"{cmd}.wav"
        started = False
        try:
            _music_proc = subprocess.Popen(
                ["mpv", "--no-video", "--audio-device=alsa/plughw:2,0",
                 "--volume=100", "--really-quiet", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
            # Give mpv/yt-dlp a second to either start streaming or fail.
            # yt-dlp failures (404, rate-limit, extractor breakage) exit fast.
            time.sleep(1.0)
            if _music_proc.poll() is None:
                log.info("MUSIC: playing %s via YouTube → Jabra", cmd)
                started = True
            else:
                log.warning("MUSIC: mpv exited %d for %s; falling back",
                    _music_proc.returncode, cmd)
                _music_proc = None
        except Exception as e:
            log.warning("MUSIC: mpv launch error %s", e)
            _music_proc = None

        if not started and local_wav.exists():
            _music_proc = subprocess.Popen(
                ["aplay", "-D", "plughw:2,0", str(local_wav)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
            log.info("MUSIC: playing %s (local wav fallback) via Jabra", cmd)
    else:
        # No URL for this cmd — go straight to local wav
        audio_file = AUDIO_DIR / f"{cmd}.wav"
        if audio_file.exists():
            _music_proc = subprocess.Popen(
                ["aplay", "-D", "plughw:2,0", str(audio_file)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
            log.info("MUSIC: playing %s (local wav) via Jabra", cmd)

    # Notify dashboard
    msg = json.dumps({"type": "music_command", "action": "stop" if cmd == "stop" else "play", "track": cmd})
    if ws_loop:
        asyncio.run_coroutine_threadsafe(broadcast(msg), ws_loop)

def _write_status():
    with lock:
        status = {
            "stage_state": stage_state,
            "stage_submode": stage_submode,
            "calendar_state": calendar_state,
            "current_class": current_class,
            "vjepa": dict(vjepa_result),
            "cameras": {cid: {"people": camera_counts.get(cid, 0), "status": camera_status.get(cid, "offline")}
                        for cid in [c["id"] for c in CAMERAS]},
            "total_people": sum(camera_counts.values()),
            "asr_active": asr_active,
            "ts": time.time(),
        }
    try:
        fd, tmp = tempfile.mkstemp(dir=STAGE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, STAGE_DIR / "brain_status.json")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# ASR thread
# ═══════════════════════════════════════════════════════════════════════════

LLM_MODEL_PATH = Path.home() / "models" / "DISABLED"
_llm = None
_llm_tried = False
_punct_buffer: list[str] = []   # raw segments waiting for punctuation
PUNCT_BATCH_SIZE = 1            # punctuate every N segments

def _load_llm():
    global _llm, _llm_tried
    if _llm_tried:
        return _llm
    _llm_tried = True
    if not LLM_MODEL_PATH.exists():
        log.warning("LLM model not found: %s", LLM_MODEL_PATH)
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        log.warning("llama_cpp not installed, skipping punctuation")
        return None
    _llm = Llama(model_path=str(LLM_MODEL_PATH), n_ctx=512, n_threads=4, verbose=False)
    log.info("LLM loaded: %s", LLM_MODEL_PATH.name)
    return _llm

def _punctuate(text: str) -> str:
    """Add punctuation to raw ASR text using Qwen2.5-7B."""
    llm = _load_llm()
    if llm is None:
        return text
    try:
        out = llm.create_chat_completion(messages=[
            {"role": "system", "content": "Add punctuation and capitalization to the raw speech transcript. Keep the original words exactly. Output only the punctuated text."},
            {"role": "user", "content": text}
        ], max_tokens=len(text) + 50, temperature=0.1)
        result = out["choices"][0]["message"]["content"].strip()
        return result if result else text
    except Exception as e:
        log.warning("Punctuation failed: %s", e)
        return text

def audio_asr_thread(loop):
    global caption_text, caption_partial, caption_segments

    if not MODEL_DIR.exists():
        log.warning("ASR model not found at %s, skipping ASR", MODEL_DIR)
        return

    # Pre-load LLM in background
    threading.Thread(target=_load_llm, daemon=True).start()

    import sherpa_onnx
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{MODEL_DIR}/tokens.txt",
        encoder=f"{MODEL_DIR}/encoder-epoch-99-avg-1.int8.onnx",
        decoder=f"{MODEL_DIR}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{MODEL_DIR}/joiner-epoch-99-avg-1.int8.onnx",
        num_threads=4, sample_rate=SAMPLE_RATE, feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=1.2,
        rule2_min_trailing_silence=0.6,
        rule3_min_utterance_length=8,
    )
    stream = recognizer.create_stream()
    block_size = int(SAMPLE_RATE * BLOCK_MS / 1000)

    proc = subprocess.Popen(
        ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
         "-c", "1", "-t", "raw", "--buffer-size", str(block_size * 4)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    log.info("ASR: listening on %s", ALSA_DEVICE)

    try:
        while not shutdown_event.is_set():
            raw = proc.stdout.read(block_size * 2)
            if not raw:
                break

            # Always process for trigger detection
            # Only caption when asr_active (checked below)

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(SAMPLE_RATE, audio)

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            current = recognizer.get_result(stream).strip()

            if recognizer.is_endpoint(stream):
                if current:
                    # Buffer raw segment, punctuate in batches
                    _punct_buffer.append(current)

                    if len(_punct_buffer) >= PUNCT_BATCH_SIZE:
                        raw_batch = " ".join(_punct_buffer)
                        _punct_buffer.clear()
                        # Show raw immediately, punctuate in background
                        with lock:
                            caption_segments.append(raw_batch)
                            if len(caption_segments) > 200:
                                caption_segments[:] = caption_segments[-200:]
                            caption_text = "\n".join(caption_segments[-50:])
                            caption_partial = ""
                        msg = json.dumps({"type": "caption", "text": caption_text, "partial": "", "segment": raw_batch})
                        try: asyncio.run_coroutine_threadsafe(broadcast(msg), loop)
                        except: pass
                        # Async punctuation: replace last segment when done
                        def _bg_punct(rb, lp):
                            p = _punctuate(rb)
                            if p != rb:
                                with lock:
                                    for i in range(len(caption_segments)-1, -1, -1):
                                        if caption_segments[i] == rb:
                                            caption_segments[i] = p
                                            break
                                    ct = "\n".join(caption_segments[-50:])
                                m = json.dumps({"type": "caption", "text": ct, "partial": "", "segment": p})
                                try: asyncio.run_coroutine_threadsafe(broadcast(m), lp)
                                except: pass
                        threading.Thread(target=_bg_punct, args=(raw_batch, loop), daemon=True).start()
                        msg = None  # already sent above
                    else:
                        # Show raw immediately, will be replaced when batch punctuates
                        with lock:
                            caption_partial = " ".join(_punct_buffer)
                        msg = json.dumps({"type": "caption", "text": caption_text, "partial": " ".join(_punct_buffer)})

                    if msg:
                        try: asyncio.run_coroutine_threadsafe(broadcast(msg), loop)
                        except: pass
                recognizer.reset(stream)
            elif current:
                with lock:
                    buf_text = " ".join(_punct_buffer + [current]) if _punct_buffer else current
                    caption_partial = buf_text
                # Voice assistant trigger detection
                for tw in TRIGGER_WORDS:
                    if tw in buf_text.upper():
                        question = buf_text.upper().split(tw, 1)[1].strip()
                        if len(question) > 5:
                            log.info("VA triggered: %s", question)
                            threading.Thread(target=_voice_assistant_respond, args=(question, loop), daemon=True).start()
                        break

                # Force-flush: if partial exceeds 80 chars, commit as segment
                if len(buf_text) > 80:
                    with lock:
                        caption_segments.append(buf_text)
                        if len(caption_segments) > 200:
                            caption_segments[:] = caption_segments[-200:]
                        caption_text = "\n".join(caption_segments[-50:])
                        caption_partial = ""
                    _punct_buffer.clear()
                    msg = json.dumps({"type": "caption", "text": caption_text, "partial": "", "segment": buf_text})
                    recognizer.reset(stream)
                else:
                    msg = json.dumps({"type": "caption", "text": caption_text, "partial": buf_text})
                try: asyncio.run_coroutine_threadsafe(broadcast(msg), loop)
                except: pass
    finally:
        # Flush remaining buffer
        if _punct_buffer:
            punctuated = _punctuate(" ".join(_punct_buffer))
            with lock:
                caption_segments.append(punctuated)
                caption_text = "\n".join(caption_segments[-50:])
            _punct_buffer.clear()
        proc.terminate()
        log.info("ASR: stopped")

# ═══════════════════════════════════════════════════════════════════════════
# WebSocket
# ═══════════════════════════════════════════════════════════════════════════

async def broadcast(message):
    if ws_clients:
        await asyncio.gather(*[c.send(message) for c in ws_clients], return_exceptions=True)

async def ws_handler(websocket):
    ws_clients.add(websocket)
    try:
        # Send current state
        await websocket.send(json.dumps({"type": "init", "state": _get_full_state()}))
        async for msg in websocket:
            try:
                data = json.loads(msg)
                if data.get("type") == "caption_toggle":
                    global asr_active
                    asr_active = bool(data.get("active", False))
                    log(f"ASR manual toggle: {asr_active}")
                    await broadcast(json.dumps({"type": "state_update", **_get_full_state()}))
            except Exception:
                pass
    finally:
        ws_clients.discard(websocket)

def _get_full_state():
    with lock:
        return {
            "stage": {"state": stage_state, "submode": stage_submode, "since": stage_state_since},
            "calendar": {"state": calendar_state, "current_class": current_class, "next_class": next_class},
            "vjepa": dict(vjepa_result),
            "cameras": {cid: {"people": camera_counts.get(cid, 0), "status": camera_status.get(cid, "offline")}
                        for cid in [c["id"] for c in CAMERAS]},
            "total_people": sum(camera_counts.values()),
            "caption": {"active": asr_active, "text": caption_text, "partial": caption_partial},
        }

async def state_broadcaster():
    """Push full state to all clients every second."""
    while not shutdown_event.is_set():
        await asyncio.sleep(1)
        if ws_clients:
            msg = json.dumps({"type": "state_update", **_get_full_state(), "ts": time.time()})
            await broadcast(msg)

# ═══════════════════════════════════════════════════════════════════════════
# HTTP server
# ═══════════════════════════════════════════════════════════════════════════

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STAGE_DIR), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path

        # Camera snapshots
        if path in ("/cam1.jpg", "/cam2.jpg", "/cam3.jpg"):
            cam_id = path[1:5]  # cam1, cam2, cam3
            with lock:
                jpeg = camera_frames.get(cam_id)
            if jpeg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(jpeg)
            else:
                self.send_error(503, "Camera not ready")
            return

        # Audio files
        if path.startswith("/audio/"):
            fname = path.split("/")[-1]
            fpath = AUDIO_DIR / fname
            if fpath.exists():
                self.send_response(200)
                ct = "audio/wav" if fname.endswith(".wav") else "audio/mpeg"
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(fpath.stat().st_size))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self.send_error(404)
            return

        # API status
        if path == "/api/status":
            data = json.dumps(_get_full_state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # Dashboard
        if path == "/" or path == "/index.html":
            self.path = "/smart_stage_dashboard.html"

        super().do_GET()

    def log_message(self, format, *args):
        pass  # suppress access logs

def start_http():
    httpd = http.server.HTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    log.info("HTTP: http://0.0.0.0:%d/", HTTP_PORT)
    httpd.serve_forever()

# ═══════════════════════════════════════════════════════════════════════════
# Discord summary
# ═══════════════════════════════════════════════════════════════════════════

def post_summary():
    transcript = "\n".join(caption_segments)
    if not transcript.strip():
        log.info("No transcript to summarize")
        return

    if DEEPSEEK_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a lecture note assistant. Summarize this transcript into structured notes."},
                    {"role": "user", "content": f"Summarize:\n\n{transcript[-30000:]}"}
                ],
                max_tokens=2000,
            )
            summary = resp.choices[0].message.content
        except Exception as e:
            summary = f"(Summary failed: {e})"
            log.warning("Summary failed: %s", e)
    else:
        summary = "(No API key, transcript only)"

    if DISCORD_WEBHOOK:
        content = f"**Smart Stage - Session Summary**\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{summary}"
        if len(content) > 1900:
            content = content[:1900] + "\n... (truncated)"
        try:
            requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=10)
            log.info("Summary pushed to Discord")
        except Exception as e:
            log.warning("Discord push failed: %s", e)

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def shutdown_handler(signum=None, frame=None):
    log.info("Shutting down...")
    shutdown_event.set()
    # Stop music
    _send_music_command("stop")
    time.sleep(2)
    if caption_segments:
        post_summary()
    sys.exit(0)

async def main():
    global ws_loop
    import websockets

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Smart Stage Brain starting...")
    log.info("Classify server: %s", CLASSIFY_URL)
    log.info("Schedule: %s", SCHEDULE_FILE)

    # Check GPU classify server
    try:
        r = requests.get(CLASSIFY_URL.replace("/classify", "/health"), timeout=5)
        log.info("GPU classify server: %s", r.json())
    except Exception as e:
        log.warning("GPU classify server not reachable: %s", e)

    ws_loop = asyncio.get_event_loop()

    # HTTP server
    http_t = threading.Thread(target=start_http, daemon=True)
    http_t.start()

    # Camera threads
    for cam in CAMERAS:
        t = threading.Thread(target=camera_thread, args=(cam,), daemon=True)
        t.start()

    # V-JEPA thread
    vj_t = threading.Thread(target=vjepa_thread, daemon=True)
    vj_t.start()

    # Brain thread
    br_t = threading.Thread(target=brain_thread, daemon=True)
    br_t.start()

    # ASR thread
    asr_t = threading.Thread(target=audio_asr_thread, args=(ws_loop,), daemon=True)
    asr_t.start()

    # State broadcaster
    asyncio.ensure_future(state_broadcaster())

    log.info("All threads started. Dashboard: http://0.0.0.0:%d/", HTTP_PORT)

    # WebSocket server
    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        log.info("WebSocket: ws://0.0.0.0:%d/", WS_PORT)
        await asyncio.Future()  # run forever



def _voice_assistant_respond(question: str, loop):
    """Gemini brain + Figurate TTS voice → play through Jabra."""
    global _va_active
    if _va_active:
        return
    _va_active = True
    try:
        # Build context from the live module-level state (brain_thread writes
        # these globals under `lock`; we just read them here).
        with lock:
            ctx_parts = []
            if current_class:
                ctx_parts.append(f"Current class: {current_class.get('name', '?')}")
            ctx_parts.append(f"Room state: {stage_submode or 'idle'}")
            ctx_parts.append(f"People: {sum(camera_counts.values()) if camera_counts else 0}")
            context = ". ".join(ctx_parts)

        # Gemini
        payload = json.dumps({
            "system_instruction": {"parts": [{"text": f"You are a friendly classroom voice assistant. Answer in 1-2 short sentences. Current room: {context}"}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"maxOutputTokens": 150, "temperature": 0.7, "thinkingConfig": {"thinkingBudget": 0}}
        }).encode()
        req = urllib.request.Request(GEMINI_URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        log.info("VA answer: %s", answer)

        # Figurate TTS
        tts_payload = json.dumps({"text": answer, "characterId": FG_CHAR_ID}).encode()
        tts_req = urllib.request.Request(FG_TTS_URL, data=tts_payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FG_API_KEY}"
        })
        with urllib.request.urlopen(tts_req, timeout=15) as resp:
            tts_data = json.loads(resp.read())

        if tts_data.get("success") and tts_data["data"].get("audio"):
            import base64, tempfile
            audio_bytes = base64.b64decode(tts_data["data"]["audio"])
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            # Play through Jabra
            subprocess.run(["mpv", "--no-video", "--audio-device=alsa/plughw:2,0",
                          "--volume=100", tmp_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            os.unlink(tmp_path)

        # Also broadcast to dashboard
        msg = json.dumps({"type": "caption", "text": f"🗣 Q: {question}\n💬 A: {answer}", "partial": "", "segment": ""})
        try:
            asyncio.run_coroutine_threadsafe(broadcast(msg), loop)
        except:
            pass
    except Exception as e:
        log.warning("VA error: %s", e)
    finally:
        _va_active = False

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, shutdown_handler)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        shutdown_handler(signum=2)


# ========== Voice Assistant (Jabra ↔ Gemini ↔ Figurate TTS) ==========
# Note: urllib.request is now imported at the top of the module so that
# _voice_assistant_respond (launched as a daemon thread from the ASR loop)
# can actually find it at runtime.
