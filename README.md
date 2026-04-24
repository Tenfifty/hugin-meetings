# hugin-meetings

Record, transcribe, diarize, and summarize meetings. Part of the
[Hugin](https://github.com/) personal productivity stack.

This is the **core engine** — language- and OS-agnostic. Frontends (tray
widgets, hotkeys, phone apps) live under `frontends/` or in separate repos.

## What it does

1. **Record** — captures mic + system audio in parallel as Opus files,
   splitting into parts at a configurable interval.
2. **Transcribe + diarize** — runs Whisper (or a fine-tuned variant) and
   a speaker diarizer, merges the outputs into a unified transcript.
3. **Match calendar event** — finds the matching Google Calendar event
   and attaches metadata (attendees, title, time, location) to the
   transcript.
4. **Summarize** — generates a structured meeting summary via a local
   LLM (llama.cpp) or a remote model (via `codex exec`).
5. **Match project/customer** — optionally links the summary to an
   existing project/customer note in your vault.
6. **Enroll speakers** — learn speaker embeddings over time so future
   recordings get named speakers instead of `SPEAKER_01`.

## Install

```
# Core (pure Python)
pip install -e .

# With transcription deps (torch, faster-whisper, pyannote.audio, pandas, numpy)
pip install -e ".[transcribe]"

# With summarization deps (openai; or install llama-cpp-python separately for local)
pip install -e ".[summarize]"
```

System dependencies (not pip-installable):

- `ffmpeg` (audio conversion)
- `codex` CLI if you want to summarize via remote models
- `gws` (Google Workspace CLI) if you want calendar matching

For the optional GNOME tray widget, install `frontends/gnome/` separately:

```
pip install -e frontends/gnome
# plus Gtk system libs: apt install python3-gi gir1.2-ayatanaappindicator3-0.1
```

## Configure

Copy `config.example.yaml` and split it into:

- `~/.config/hugin/hugin.yaml`   — shared across all hugin-* tools
- `~/.config/hugin/meetings.yaml` — meetings-specific

`meetings.yaml` overrides `hugin.yaml`. Every field has a sensible default
(English, generic summary prompt, matcher disabled), so a minimal config is fine.

Override the config dir with `HUGIN_CONFIG_DIR=/path`.

## CLI

After `pip install -e .`:

| Command | What |
|---------|------|
| `hugin-meet-transcribe [session-id\|file]` | Transcribe + diarize a recording session |
| `hugin-meet-summarize [transcript]` | Summarize a transcript |
| `hugin-meet-match-calendar [transcript]` | Attach calendar metadata |
| `hugin-meet-enroll` | Interactively enroll a new speaker |
| `hugin-meet-compare-diarization` | Benchmark diarization pipelines |
| `hugin-meet-tui` | Interactive curses TUI — drives the whole pipeline |

## Writing a frontend

The GNOME tray widget lives in `frontends/gnome/` as a separate installable
package. Port it to your OS/desktop by copying that directory and replacing
the Gtk layer; the core engine (`hugin_meetings.*`) stays the same.

Key integration points a frontend needs:

- Spawn `ffmpeg` to write to `cfg.raw_audio_dir / {mic|sys}-{session_id}-p{NN}.opus`
- Call `hugin-meet-transcribe <session-id>` when done
- Read pipeline state via `hugin_meetings.pipeline.scan_raw_audio_sessions()`

## Layout

```
src/hugin_meetings/
  config.py               Config loader (hugin.yaml + meetings.yaml)
  pipeline.py             Metadata, matching, shared pipeline helpers
  transcribe.py           Whisper + diarization
  transcribe_part.py      Per-part worker (spawned as subprocess)
  summarize.py            LLM summarization
  enroll.py               Speaker enrollment
  calendar_match.py       Google Calendar matching
  diarization_compare.py  Diarization benchmark
  tui.py                  Interactive TUI
  prompts/                Summary prompt templates

frontends/
  gnome/                  GNOME tray widget (separate install target)
```

## Status

Early. Config boundary stable, internals still in flux. MIT licensed.
