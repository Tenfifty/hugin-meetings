# Repo guidance for Claude / Codex

`hugin-meetings` is the **core engine** of a meeting recorder /
transcriber / summarizer pipeline — part of the [Hugin](../hugin)
personal productivity stack. It is deliberately OS- and language-
agnostic. GUI/tray frontends (e.g. `frontends/gnome/`) are separate
installable packages that call into this engine via CLI entry points
and read pipeline state by scanning directories.

The shared contract (config layout, language handling, vault
structure, markdown headers, LLM provider naming, prompt-file
convention) lives in [`../hugin/CONVENTIONS.md`](../hugin/CONVENTIONS.md).
Read that before touching `config.py` or anything that crosses tool
boundaries.

## Install / dev setup

```
pip install -e . --user --break-system-packages
pip install -e ".[transcribe,summarize-local]"   # optional extras
```

The `--break-system-packages` flag is required on PEP 668 systems
(Ubuntu/Debian). `hugin` (the shared library) is pulled in
transitively; install it editable too if you want to develop against
local hugin changes:

```
pip install -e ~/projs/hugin --user --break-system-packages
```

System deps (not pip-installable): `ffmpeg` (audio), `codex` / `claude` /
`gemini` CLI (remote summary providers), `gws` (Google Workspace CLI,
for calendar matching). `torch`/`pyannote.audio` come in via the
`transcribe` extra.

## Tests

```
pytest                                 # full suite
pytest tests/test_pipeline_delete.py   # single file
pytest tests/test_pipeline_delete.py::test_name   # single test
```

Tests that touch config use `reset_config_cache()` from `hugin_meetings.config` because `load_config()` is `lru_cache`d — see `tests/test_prompts_config.py` for the pattern.

## CLI entry points (installed by `pip install -e .`)

All are defined in `pyproject.toml [project.scripts]`:

- `hugin-meet-transcribe` → `transcribe:main` — Whisper + diarization, merges mic+sys tracks
- `hugin-meet-summarize` → `summarize:main` — LLM summarization (local llama.cpp OR `codex exec` remote)
- `hugin-meet-match-calendar` → `calendar_match:main` — attaches Google Calendar metadata via `gws`
- `hugin-meet-enroll` → `enroll:main` — interactive speaker enrollment
- `hugin-meet-compare-diarization` → `diarization_compare:main`
- `hugin-meet-tui` → `tui:main` — curses driver for the whole pipeline

## Architecture

The pipeline is **file/directory driven, not daemon-driven**. Each stage reads from and writes to well-known directories; state discovery works by scanning them. This is why frontends and the TUI can coordinate without IPC.

```
raw opus files  →  transcribe.py (Whisper + diarizer + merge)  →  transcript .md
                                                               →  transcripts/*.json (cache)
transcript      →  calendar_match.py (gws → event metadata block)
transcript      →  summarize.py (LLM) → summaries/*.md
summary         →  pipeline.py project/customer matcher → links summary into a project note
```

### Key modules

- **`config.py`** — `MeetingsConfig` subclasses `hugin.SharedConfig`. Loading goes through `hugin.config.load_tool("meetings", MeetingsConfig.from_merged)`, which reads `~/.config/hugin/hugin.yaml` + `meetings.yaml` and deep-merges. `HUGIN_CONFIG_DIR` overrides the dir. `load_config()` is `lru_cache(maxsize=1)`. `MeetingsConfig` exposes `state_dir` and derived subdirs (`raw_audio_dir`, `wav_cache_dir`, `speakers_dir`, `models_dir`, `transcript_json_dir`, `recorder_state_dir`). The `LLMConfig`, the codex/claude/gemini runner (`hugin.llm.run_prompt`), and the prompt resolver (`hugin.prompts.resolve_prompt`, which auto-picks `<base>_<lang>.md`) all live in the shared library.
- **`pipeline.py`** — central metadata/filename conventions and shared helpers. Most cross-module logic lives here: session scanning (`scan_raw_audio_sessions`, `scan_recordings`), filename parsing, calendar metadata markers, project/customer matching. When adding a new pipeline stage, read this first.
- **`transcribe.py`** / **`transcribe_part.py`** — `transcribe_part.py` is spawned as a **subprocess** per audio part (to release GPU memory between parts). Do not refactor that into an in-process call without thinking about VRAM.
- **`summarize.py`** — dispatches to either local llama.cpp models (via `LOCAL_MODELS`) or the shared `hugin.llm.run_prompt` for codex / claude / gemini. Prompt selection uses `hugin.prompts.resolve_prompt` so `language: sv` auto-picks `prompts/summary_sv.md` when shipped.
- **`calendar_match.py`** — shells out to `gws`. By default only searches calendars the user owns; `--include-shared-calendars` / `--calendar <id>` override.
- **`tui.py`** — curses UI that orchestrates the other CLIs; the canonical example of how a frontend should drive the pipeline.

### Filename / directory conventions (load-bearing — see `pipeline.py`)

- Raw audio: `{mic|sys}-{YYYYMMDD-HHMMSS}-p{NN}.opus` in `cfg.raw_audio_dir`. Parsed by `RAW_AUDIO_RE`. Session ID is the timestamp.
- Transcripts: `transcripts_dir/transcript-{ts}.md`, JSON cache in `cfg.transcript_json_dir`.
- Summaries: `summaries_dir/*.md`.
- Calendar metadata in transcripts is bracketed by `<!-- calendar-metadata:start -->` / `<!-- calendar-metadata:end -->` (constants `CALENDAR_METADATA_START/END`).
- Summary header is configurable (`summary_header`, default `## Meeting Summary`); `personal_section_header` optionally carves out an H3 for personal follow-ups.
- Speaker labels in transcripts match `SPEAKER_RE` (`speaker_01`, `SPEAKER_01`, optional `_p01` part suffix).

### Prompts

Summary + project-matcher prompts are plain Markdown templates in `src/hugin_meetings/prompts/` (shipped as package data). Resolution order — see `hugin.prompts.resolve_prompt`:

1. Explicit `meetings.summarize_prompt_path` / `meetings.project_matcher.prompt_path` in config.
2. `<base>_<lang>.md` for the active language (e.g. `summary_sv.md`).
3. `<base>_default.md` (English fallback, always shipped).

Files suffixed `.example.md` are starter templates for users to copy — they are never auto-picked. Matcher templates interpolate `{{candidate_context}}`, `{{calendar_lines}}`, `{{summary_body}}`, `{{internal_rules}}`.

### Frontend integration contract

A frontend's job is narrow:

1. Spawn `ffmpeg` to write `{mic|sys}-{session_id}-p{NN}.opus` into `cfg.raw_audio_dir`.
2. Call `hugin-meet-transcribe <session-id>` when done.
3. Read state via `hugin_meetings.pipeline.scan_raw_audio_sessions()`.

Don't expand this surface casually — frontends live in other repos and would break.

## Status

Early. Config boundary is stable; internals are still in flux.
