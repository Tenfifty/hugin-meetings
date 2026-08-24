# Repo guidance for Claude / Codex

`hugin-meetings` is the **core engine** of a meeting recorder /
transcriber / summarizer pipeline — part of the Hugin
personal productivity stack. It is deliberately OS- and language-
agnostic. GUI/tray frontends (e.g. `frontends/gnome/`) are separate
installable packages that call into this engine via CLI entry points
and read pipeline state by scanning directories.

The shared contract (config layout, language handling, vault
structure, markdown headers, LLM provider naming, prompt-file
convention) lives in the separate hugin package in `hugin/CONVENTIONS.md`.
Read that before touching `config.py` or anything that crosses tool
boundaries.

## Install / dev setup

```
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[transcribe,summarize-local]"   # optional extras
```

Use the repo-local venv for the transcribe stack. WhisperX, PyAnnote,
NeMo, Torch, NumPy, Pandas, and pip-installed NVIDIA CUDA/cuDNN runtimes
have tight binary compatibility constraints, and global/user-site
installs on PEP 668 systems can conflict with unrelated tools. `hugin`
(the shared library) is pulled in transitively; install it editable too
if you want to develop against local hugin changes:

```
python -m pip install -e ~/projs/hugin
python -m pip install -e ".[transcribe]" -e frontends/gnome
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

- `hugin-meet-context` → `context:main` — **first stage.** Guesses language (`langid.py`), calendar event and customer for a session, before anything expensive runs
- `hugin-meet-langid` → `langid:main` — the language probe on its own, useful for checking a batch of sessions
- `hugin-meet-transcribe` → `transcribe:main` — Whisper + diarization, merges mic+sys tracks
- `hugin-meet-summarize` → `summarize:main` — LLM summarization (local llama.cpp OR `codex exec` remote)
- `hugin-meet-match-calendar` → `calendar_match:main` — attaches Google Calendar metadata via `gws`
- `hugin-meet-enroll` → `enroll:main` — interactive speaker enrollment
- `hugin-meet-link` → `link:main` — **last stage.** Publishes a summary into its verified customer note
- `hugin-meet-tui` → `tui:main` — curses driver for the whole pipeline

## Architecture

The pipeline is **file/directory driven, not daemon-driven**. Each stage reads from and writes to well-known directories; state discovery works by scanning them. This is why frontends and the TUI can coordinate without IPC.

```
recording ends →  context.py  →  context-{ts}.json   language (langid) + calendar event + customer
                                        ↓
                              [ a person verifies ]   TUI only; nothing expensive runs before this
                                        ↓
raw opus files  →  transcribe.py (language from context; Whisper + diarizer + merge)
                                                               →  transcript .md
                                                               →  transcripts/*.json (cache)
transcript      →  calendar_match.py (renders the stored event; --rematch to ask gws again)
transcript      →  summarize.py (LLM) → summaries/*.md
summary         →  link.py → writes the summary into the verified customer note
```

**Verification is a precondition, not a correction.** The language and the
customer are *inputs* to the pipeline, decided up front and confirmed by a
person, not conclusions drawn from its output afterwards. Two rules follow:
`transcribe` refuses to run without a context, and nothing infers a customer
from a summary — that would only contradict the person who already decided.
The TUI additionally refuses to process a session that is not verified;
the engine only requires that the context exists.

### Key modules

- **`config.py`** — `MeetingsConfig` subclasses `hugin.SharedConfig`. Loading goes through `hugin.config.load_tool("meetings", MeetingsConfig.from_merged)`, which reads `~/.config/hugin/hugin.yaml` + `meetings.yaml` and deep-merges. `HUGIN_CONFIG_DIR` overrides the dir. `load_config()` is `lru_cache(maxsize=1)`. `MeetingsConfig` exposes `state_dir` and derived subdirs (`raw_audio_dir`, `wav_cache_dir`, `speakers_dir`, `models_dir`, `transcript_json_dir`, `recorder_state_dir`). The `LLMConfig`, the codex/claude/gemini runner (`hugin.llm.run_prompt`), and the prompt resolver (`hugin.prompts.resolve_prompt`, which auto-picks `<base>_<lang>.md`) all live in the shared library.
- **`pipeline.py`** — central metadata/filename conventions and shared helpers. Most cross-module logic lives here: session scanning (`scan_raw_audio_sessions`, `scan_recordings`), filename parsing, calendar metadata markers, customer notes and the matcher prompt. When adding a new pipeline stage, read this first. `MeetingStatus` counts five steps — verify, transcribe, calendar, summarize, link — and `ready_for_pipeline` is false until a person has verified.
- **`context.py`** — the first stage. `build_context()` guesses, `verify()` is the single write point for a person's decision (it also materializes the customer note and writes `.customer.json`), `record_applied()` notes what a stage actually used so a later change reads as drift. Imports `pipeline`, so `pipeline` may only import it inside functions.
- **`langid.py`** — VAD-filtered language identification. Samples three 30s windows at 25/50/75% of the *speech* of each channel's largest part and votes. Never samples the opening: meetings routinely start in the local language while waiting for a guest who switches the room to English. Only the sampled regions are decoded, and the probes run as threads against one shared `base` CPU model.
- **`transcribe.py`** / **`transcribe_part.py`** — `transcribe_part.py` is spawned as a **subprocess** per audio part (to release GPU memory between parts). Do not refactor that into an in-process call without thinking about VRAM.
- **`summarize.py`** — dispatches to either local llama.cpp models (via `LOCAL_MODELS`) or the shared `hugin.llm.run_prompt` for codex / claude / gemini. Prompt selection uses `hugin.prompts.resolve_prompt` so `language: sv` auto-picks `prompts/summary_sv.md` when shipped.
- **`calendar_match.py`** — shells out to `gws`. By default only searches calendars the user owns; `--include-shared-calendars` / `--calendar <id>` override. Matching works from a `MatchWindow`, which comes either from a transcript or straight from raw audio — that is what lets the calendar be matched before transcription. Candidates are stored whole in the context (`serialize_candidate`) so the metadata block and the matcher prompt share one representation.
- **`link.py`** — the last stage. Writes the customer link into the summary and the summary into the customer note. Refuses unverified state.
- **`tui.py`** — curses UI that orchestrates the other CLIs. It is the pipeline driver: frontends hand off to it rather than calling the stage CLIs themselves (see *Frontend integration contract*).

### Filename / directory conventions (load-bearing — see `pipeline.py`)

- Per-session files live under a `YYYY/` subdir of their base dir, derived from the session timestamp by `pipeline.year_subdir()`. Scans use `rglob`; writers must go through the path helpers.
- Raw audio: `cfg.raw_audio_dir/YYYY/{mic|sys}-{YYYYMMDD-HHMMSS}-p{NN}.opus`, built by `recording.raw_audio_part_path`. Parsed by `RAW_AUDIO_RE`. Session ID is the timestamp.
- Transcripts: `transcripts_dir/YYYY/transcript-{ts}.md`, JSON cache at `cfg.transcript_json_dir/YYYY/transcript-{ts}.json` (`pipeline.transcript_json_path`). The JSON is a bare list of segment dicts — several modules read it positionally, so it has no metadata wrapper.
- Summaries: `summaries_dir/YYYY/summary-{ts}.md`.
- Customer/project match state: `cfg.transcript_json_dir/YYYY/transcript-{ts}.customer.json` (`pipeline.customer_state_path`).
- Calendar metadata in transcripts is bracketed by `<!-- calendar-metadata:start -->` / `<!-- calendar-metadata:end -->` (constants `CALENDAR_METADATA_START/END`).
- Summary header is configurable (`summary_header`, default `## Meeting Summary`); `personal_section_header` optionally carves out an H3 for personal follow-ups.
- Speaker labels in transcripts match `SPEAKER_RE` (`speaker_01`, `SPEAKER_01`, optional `_p01` part suffix).

### Prompts

Summary + project-matcher prompts are plain Markdown templates in `src/hugin_meetings/prompts/` (shipped as package data). Resolution order — see `hugin.prompts.resolve_prompt`:

1. Explicit `meetings.summarize_prompt_path` / `meetings.project_matcher.prompt_path` in config.
2. `<base>_<lang>.md` for the active language (e.g. `summary_sv.md`).
3. `<base>_default.md` (English fallback, always shipped).

Files suffixed `.example.md` are starter templates for users to copy — they are never auto-picked. Matcher templates interpolate `{{candidate_context}}`, `{{calendar_lines}}`, `{{summary_body}}`, `{{internal_rules}}`, `{{evidence_rules}}`.

### Frontend integration contract

A frontend's job is narrow: **record, and show state**. It does not run the
pipeline — it hands off to the TUI, which drives transcription/summarization.
`frontends/gnome/` is the reference implementation of exactly this surface.

1. Record via `hugin_meetings.recording` — `RecordingSession(audio_dir=cfg.raw_audio_dir)`
   with `start()` / `stop()` / `rotate()`. It builds the `ffmpeg` command and
   writes parts to the right path (`recording.raw_audio_part_path`, which
   includes the `YYYY/` subdir). Don't reimplement the ffmpeg invocation or the
   filename layout in the frontend.
2. Resolve input devices via `hugin_meetings.audio_routes`
   (`get_default_audio_routes()`), not by parsing PipeWire yourself.
3. Read pipeline state via `hugin_meetings.pipeline.scan_recordings()` →
   `MeetingStatus.needs_pipeline` for a pending count.
   `scan_raw_audio_sessions()` is the rawer view if that is all you need.
4. Hand off processing by launching `hugin-meet-tui`
   (`cli_utils.resolve_sibling_bin("hugin-meet-tui")`). Frontends do **not**
   call `hugin-meet-transcribe` / `-summarize` / `-match-calendar` directly;
   those stages have ordering requirements the TUI owns.
5. Optionally surface upcoming meetings via `hugin_meetings.schedule`
   (journal-derived, plus the recorder reminder state).

Don't expand this surface casually — frontends live in other repos and would break.

## Status

Early. Config boundary is stable; internals are still in flux.
