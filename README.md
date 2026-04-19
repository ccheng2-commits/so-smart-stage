# so-smart-stage — Dual-Layer Classroom Orchestrator

Raspberry Pi 5 orchestrator that reads a classroom schedule and a live V-JEPA video classification on a single shared event loop, and drives three actions in lockstep: ambient music that shifts with room phase, live lecture subtitles with a DeepSeek summary pushed to Discord on shutdown, and a wake-word voice assistant that answers in a synthesized voice.

**What's Included:**
- 🧠 `src/smart_stage_brain.py` — ~990-line threaded orchestrator (3 × OAK-D-POE cameras, V-JEPA HTTP client, calendar × V-JEPA state machine, sherpa-onnx streaming ASR, Gemini + Figurate TTS voice assistant, WebSocket + HTTP server)
- 📺 `static/smart_stage_dashboard.html` — live 3-camera tile view, state machine visualization, synced captions
- 📝 `static/live_caption.html` — full-screen caption display for the class screen
- 📅 `schedule.example.json` — schedule schema (the real file lives at `~/smart-stage/schedule.json` and is gitignored)
- 🔔 `audio/chime.wav` — confirmation chime
- 🎙️ `models/download.sh` — fetches the 260 MB sherpa-onnx streaming Zipformer bilingual ASR model

---

## 🧩 Relationship to `kandizzy/smart-objects-cameras`

**Classification: Companion, downstream of so-vjepa-probe.**

The course prompt originally framed three separate extensions — overhead dashboard (#1), room-phase music controller (#3), live lecture subtitle generator (#4). In this implementation they are **one orchestrator**. Music choices, subtitle triggers, and dashboard tiles all consume the same `calendar_state × vjepa_class` state machine; splitting them would mean duplicating that state machine across three processes or inventing a new IPC just to share it. Combining is the honest call, and the README flags it so reviewers can see the decision was deliberate.

Where it aligns with the template's patterns, and where it doesn't:

| Template pattern | This repo |
|---|---|
| Detectors write `~/oak-projects/camera_status.json` + `latest_frame.jpg` | ❌ Does not read or write the template IPC files. Each camera thread opens its own `depthai.Device(ip)` and publishes JPEG bytes + detection counts into in-process dicts. The brain writes its own consolidated state to `~/smart-stage/brain_status.json`. |
| `--discord` / `--log` CLI flags | ❌ Discord posting is unconditional (on shutdown) and controlled by `DISCORD_WEBHOOK_URL` env; there is no CLI toggle. |
| Temporal debouncing | ✅ 15-second `MODE_DEBOUNCE` on state transitions, in the same spirit as the template's smoothing. |
| Config polling for dynamic reconfigure | ⚠️ `schedule.json` is reloaded every 60 s; camera IPs, classify URL, ASR model path are start-time only. |
| DepthAI 3.x pipeline style | ✅ DepthAI 3.x with `NNArchive(getModelFromZoo(...))`. |

The brain is **strictly downstream of [so-vjepa-probe](https://github.com/ccheng2-commits/so-vjepa-probe)**: every 10 s, a 3-second clip from the camera with the most people is POSTed to `http://<gpu>:8766/classify` and the returned class label drives music and submode decisions. If the classify server is unreachable, music stays in whatever state the last successful classification picked; the rest of the system keeps running.

---

## 🧰 Hardware & dependencies

- Raspberry Pi 5 (16 GB), Debian 13, Python 3.13
- 3 × Luxonis OAK-D-POE on the `169.254.1.x` link-local segment
- USB microphone on ALSA `hw:2,0` (for ASR)
- Jabra USB speakerphone on ALSA `plughw:2,0` (for music + TTS output)
- GPU host running [so-vjepa-probe](https://github.com/ccheng2-commits/so-vjepa-probe) reachable at the IP in `CLASSIFY_URL`
- Python: `depthai >= 3.x`, `opencv-python`, `numpy`, `requests`, `sherpa-onnx`, `websockets`, `openai` (DeepSeek compat)
- System: `mpv`, `yt-dlp`, `aplay` (from `alsa-utils`), `arecord`

No cloud service is required for the core dashboard + subtitle flow. The voice assistant and summary features are optional — if the corresponding env vars are blank, those paths silently no-op.

---

## 🚀 Setup

```bash
# On the Pi
git clone https://github.com/ccheng2-commits/so-smart-stage.git
cd so-smart-stage

# 1. ASR model (~260 MB)
bash models/download.sh
# Symlink so the brain can find it at ~/smart-stage/models/:
mkdir -p ~/smart-stage/models
ln -s "$(pwd)/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20" \
      ~/smart-stage/models/

# 2. Audio fallback wavs (see audio/README.md)
# chime.wav is already present; provide welcome.wav / bg_light.wav / farewell.wav.
mkdir -p ~/smart-stage/audio
cp audio/*.wav ~/smart-stage/audio/

# 3. Deploy runtime files to ~/smart-stage/
# The brain's HTTP handler serves from ~/smart-stage/ (STAGE_DIR), so the HTMLs
# must live there:
cp src/smart_stage_brain.py     ~/smart-stage/
cp static/smart_stage_dashboard.html ~/smart-stage/
cp static/live_caption.html     ~/smart-stage/
cp schedule.example.json        ~/smart-stage/schedule.json   # edit to match your actual schedule

# 4. Secrets
sudo cp .env.example /etc/smart-stage.env
sudo chmod 600 /etc/smart-stage.env
sudo nano /etc/smart-stage.env    # fill in DEEPSEEK_API_KEY, DISCORD_WEBHOOK_URL, GEMINI_API_KEY, FIGURATE_*

# 5. Dashboard client-side keys (optional, only if you want the browser
#    to call Figurate/Gemini directly — see the big caveat in Known Limitations).
#    Before the <script> tag in smart_stage_dashboard.html, add something like:
#      <script>
#        window.FG_KEY = 'fg_...';
#        window.FG_CHAR = 'cmn...';
#        window.GEMINI_KEY = 'AIza...';
#      </script>
#    If unset, the brain-side voice assistant still works end-to-end; only
#    browser-local TTS previews will be disabled.
```

## ▶️ Usage

```bash
# On the Pi, from anywhere:
python3 ~/smart-stage/smart_stage_brain.py

# Then from any host on the network / Tailscale:
open http://<pi-host>:8090/smart_stage_dashboard.html
# or for the classroom big screen:
open http://<pi-host>:8090/live_caption.html
```

Graceful shutdown (Ctrl-C or SIGTERM) triggers `post_summary()`, which feeds the captured transcript to DeepSeek and posts the summary to the Discord webhook.

---

## 🔧 What's fixed in this repo vs the Pi's original

Three targeted fixes were applied before first commit; they're isolated into their own commits so the diffs are reviewable:

1. **Secrets moved from source to environment** (`commit 95e412a`): `GEMINI_API_KEY`, `FIGURATE_API_KEY`, and `FIGURATE_CHAR_ID` were hardcoded as string literals in `smart_stage_brain.py` and in the dashboard HTML. The brain now reads them from `/etc/smart-stage.env` via the existing `_load_env()` helper, and the HTML reads them from `window.*` globals (which you set in a private inline script — see setup step 5).

2. **Voice-assistant context builder repaired** (`commit ceb78a5`): `_voice_assistant_respond` referenced a `brain_status` dict that was never defined anywhere, plus four bare identifiers that Python treated as undefined names. Every wake-word trigger threw `NameError` and the exception was swallowed, so the 2026-04-13 demo logs showed "trigger detected, no response" — this was the cause. Now reads `current_class`, `stage_submode`, and `camera_counts` directly, all of which the brain thread maintains under `lock`.

3. **Local wav fallback for music** (`commit 39d4c72`): `MUSIC_URLS` points at three public YouTube streams (Lofi Girl + piano). `mpv` + `yt-dlp` breaks regularly because YouTube rotates extractor signatures. The brain now waits 1 s after launching mpv; if the process exits (the yt-dlp failure signal), it falls back to `~/smart-stage/audio/{cmd}.wav`. Operator-provided `welcome.wav` / `bg_light.wav` / `farewell.wav` (see `audio/README.md`) keep the class from going silent when YouTube misbehaves.

---

## ⚠️ Known limitations

- **Client-side API keys in the dashboard.** The current dashboard HTML calls Gemini and Figurate directly from the browser, which means any client holding the page gets the keys. Server-side proxying through the brain would be the right refactor but isn't in this repo yet. For now, the HTML defaults the keys to empty strings; set `window.*` globals in a private inline script if you want browser-side voice features. The brain-side voice assistant does not depend on this.
- **YouTube as an ambient music source is fragile.** yt-dlp extractors break every few weeks; the wav fallback exists precisely because of this. Treat the YouTube URLs as demo placeholders, not production.
- **LLM punctuation is disabled by default.** `LLM_MODEL_PATH = ~/models/DISABLED` deliberately points at a path that doesn't exist, so the ASR captions are raw sherpa-onnx output without capitalization or punctuation. Point `_load_llm()` at an actual llama.cpp GGUF to enable; expect ~200–500 ms additional latency per finalized segment.
- **Voice assistant fires on any partial ASR match of `HEY SMART OBJECT`, `SMART OBJECT`, or `HEY SMART OBJECTS`** in the currently-buffered utterance. False triggers from the teacher saying "smart object" as a noun are common. A proper wake-word detector (Porcupine, openWakeWord) would be safer.
- **No authentication on port 8090 or 8091.** Binds `0.0.0.0`. Run on the classroom LAN or behind Tailscale.
- **Hardcoded hardware assumptions:** camera IPs (`169.254.1.{10,11,222}`), classify URL (`100.113.55.109:8766`), ASR mic (`hw:2,0`), speaker (`plughw:2,0`). Change the constants at the top of `smart_stage_brain.py` for other setups.
- **The HTTP server serves files from `STAGE_DIR` (i.e. `~/smart-stage/`), not from the repo root.** That's why the setup steps `cp` things into `~/smart-stage/` instead of running from the clone directory.
- **On shutdown, the DeepSeek summarizer is called with the full accumulated transcript.** For a multi-hour class this trims to 30,000 chars; longer sessions lose the earliest content.

---

## 🔗 Links

- Class template: [kandizzy/smart-objects-cameras](https://github.com/kandizzy/smart-objects-cameras)
- Sibling extensions:
  - [so-overhead-dashboard](https://github.com/ccheng2-commits/so-overhead-dashboard) — bird's-eye ground-plane fusion of the same three cameras
  - [so-vjepa-probe](https://github.com/ccheng2-commits/so-vjepa-probe) — the 4-class V-JEPA classifier this repo depends on
