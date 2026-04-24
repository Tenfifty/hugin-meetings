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
   LLM (llama.cpp) or a remote model through Codex, Claude Code, or Gemini CLI.
5. **Match project/customer** — optionally links the summary to an
   existing project/customer note in your vault.
6. **Enroll speakers** — learn speaker embeddings over time so future
   recordings get named speakers instead of `SPEAKER_01`.

## Install

```
# Core (pure Python)
pip install -e .

# With transcription and speaker-enrollment deps
pip install -e ".[transcribe]"

# Remote summarization through codex/claude/gemini uses CLI tools, no Python extra.

# With local llama.cpp summarization deps
pip install -e ".[summarize-local]"
```

System dependencies (not pip-installable):

- `ffmpeg` (audio conversion)
- `codex`, `claude`, or `gemini` CLI if you want to summarize via remote models
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

The summary and project/customer matcher prompts are ordinary Markdown
templates under `src/hugin_meetings/prompts/`. Set
`meetings.summarize_prompt_path` or `meetings.project_matcher.prompt_path` to
use your own versions. Matcher templates can use `{{candidate_context}}`,
`{{calendar_lines}}`, `{{summary_body}}`, and `{{internal_rules}}`.

Remote and local command models use `meetings.llm.provider`, which can be
`codex`, `claude`, `gemini`, or `local`. The default is `codex`. Claude Code
runs from a clean working directory so repo-local `CLAUDE.md` files are not
discovered while normal Claude Code login still works. Gemini also runs from a
clean working directory, with context discovery pointed at a missing file.

Set `summary_model` or `project_matcher.model` to `default` to let the provider
CLI choose its configured model. Summaries default to `summary_effort: high`;
project matching defaults to `project_matcher.effort: low`. Effort is applied
for Codex and Claude, and ignored for Gemini.

For local one-shot inference, use `meetings.llm.provider: local` and set
`meetings.llm.local_command`. The prompt is written to stdin and stdout is used
as the model response, which works well for llama.cpp-style commands that should
exit and release GPU memory after each request.

## CLI

After `pip install -e .`:

| Command | What |
|---------|------|
| `hugin-meet-transcribe [session-id\|file]` | Transcribe + diarize a recording session |
| `hugin-meet-summarize [transcript]` | Summarize a transcript |
| `hugin-meet-match-calendar [transcript]` | Attach calendar metadata |
| `hugin-meet-enroll` | Interactively enroll a new speaker |
| `hugin-meet-tui` | Interactive curses TUI — drives the whole pipeline |

Calendar matching searches calendars you own by default, which keeps shared or
subscribed calendars from being picked accidentally. Use
`hugin-meet-match-calendar --calendar <id>` for one specific calendar, or
`--include-shared-calendars` if shared calendars should be considered too.

## Writing a frontend

The GNOME tray widget lives in `frontends/gnome/` as a separate installable
package. Port it to your OS/desktop by copying that directory and replacing
the Gtk layer; the core engine (`hugin_meetings.*`) stays the same.

Key integration points a frontend needs:

- Use `hugin_meetings.recording` to write `{mic|sys}-{session_id}-p{NN}.opus`
  into `cfg.raw_audio_dir`
- Use `hugin_meetings.audio_routes` on Linux/PipeWire to discover mic/system
  audio routes, or provide an OS-specific route provider
- Use `hugin_meetings.schedule` for journal meeting parsing and reminder state
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
  tui.py                  Interactive TUI
  prompts/                Summary and matcher prompt templates

frontends/
  gnome/                  GNOME tray widget (separate install target)
```

## Status

Early. Config boundary stable, internals still in flux. MIT licensed.
