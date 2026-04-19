# audio/

Local playback fallback files for `smart_stage_brain.py`. The brain tries the
hardcoded YouTube URL in `MUSIC_URLS` first and falls back to a wav file in
this directory if `mpv` + `yt-dlp` fails.

## Expected files

| File | Used by state | Notes |
|---|---|---|
| `chime.wav` | `chime` command | **Committed.** Short confirmation chime, plays on trigger events. |
| `welcome.wav` | pre-class, people present | **Not committed** (`.gitignore` excludes it). Provide your own. |
| `bg_light.wav` | in-class, V-JEPA says `group_work` | **Not committed.** Provide your own. |
| `farewell.wav` | class ending | **Not committed.** Provide your own. |

## Populating the three missing files

Any ambient wav works. The brain just calls `aplay -D plughw:2,0 <file>`.
Durations from a few seconds up to a full class all play fine (mpv and
aplay both stay on a single track, no playlist continuation).

One option, if you want to mirror the YouTube fallback behavior exactly:

```bash
cd audio/
yt-dlp -x --audio-format wav -o welcome.wav  "https://www.youtube.com/watch?v=jfKfPfyJRdk"
yt-dlp -x --audio-format wav -o bg_light.wav "https://www.youtube.com/watch?v=rUxyKA_-grg"
yt-dlp -x --audio-format wav -o farewell.wav "https://www.youtube.com/watch?v=1fueZCTYkpA"
```

Respect the original licenses of whatever you download. The source files are
not committed here on purpose.

## ALSA device

The brain hardcodes `plughw:2,0`, matching a Jabra USB speakerphone. If your
output device is different, edit the `aplay` / `mpv --audio-device=...` lines
in `src/smart_stage_brain.py`.
