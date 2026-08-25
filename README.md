# hugin-meetings

Record, transcribe, diarize, and summarize meetings. Part of the
[Hugin](https://github.com/Tenfifty/hugin) personal productivity stack.

This is the **core engine** — language- and OS-agnostic. Frontends (tray
widgets, phone apps) live under `frontends/` or in separate repos.

## What it does

1. **Record** — captures mic + system audio in parallel as Opus files,
   splitting into parts at a configurable interval.
2. **Guess the context** — as soon as recording stops, works out what the
   meeting *was*: the spoken language (sampled from the audio), the matching
   Google Calendar event, and which project/customer note it belongs to.
3. **Verify** — you confirm or correct language and customer in the TUI.
   Nothing expensive runs before this, and the answers are then used as
   *inputs* by the stages below rather than guessed again by each of them.
4. **Transcribe + diarize** — runs Whisper (or a fine-tuned variant) in the
   confirmed language, plus a speaker diarizer, and merges the outputs into a
   unified transcript with the calendar metadata attached.
5. **Summarize** — generates a structured meeting summary via a local
   LLM (llama.cpp) or a remote model through Codex, Claude Code, or Google
   Antigravity (`agy`).
6. **Publish** — writes the summary into the linked project/customer note.
7. **Post to Slack** — posts the meeting abstract to a project Slack
   channel, with the full summary (minus personal notes) as a thread reply.
8. **Enroll speakers** — learn speaker embeddings over time so future
   recordings get named speakers instead of `SPEAKER_01`.

Why the guess comes first: Whisper detects language from the first 30 seconds
of a recording, which on a meeting that opens with people connecting is 30
seconds of empty room — that is how a Swedish meeting once got transcribed as
Dutch. And a customer guessed from a finished summary can only ever contradict
the person who already knows the answer. Both questions are answerable the
moment recording stops.

The features are driven from command line utilities, but primary usage is expected to be through the TUI, `hugin-meet-tui` and optionally a desktop integration with a widget.

The frontend reminder for starting/stopping recording, polls your `journal.md` and reacts on any entry of the form
`- [ ] Meeting about X *{16:00 - 16:30}*`

These entries will be manually written, or automatically synced from Google Calendar, see [hugin-agenda]('https://github.com/Tenfifty/hugin-agenda')

To not be reminded on a timed entry, write "~" before the time:
`- [ ] Fika *~{15:00}*`

## Install

```
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

# Core (pure Python)
python -m pip install -e .

# With transcription and speaker-enrollment deps
python -m pip install -e ".[transcribe]"

# Remote summarization through codex/claude/agy uses CLI tools, no Python extra.

# With local llama.cpp summarization deps
python -m pip install -e ".[summarize-local]"
```

When developing from sibling checkouts, install the shared `hugin` library
editable first, then this repo, then any frontend package.

```
python -m pip install -e ~/projs/hugin
python -m pip install -e ".[transcribe]" -e frontends/gnome
```

Use the repo-local virtualenv for the audio stack. WhisperX, PyAnnote, NeMo,
Torch, NumPy, and Pandas have tight binary compatibility constraints, and a
global/user-site Python install can easily conflict with unrelated tools.
The `transcribe` extra also installs the pip NVIDIA runtime libraries needed
by the CUDA model backends.

Run commands with the venv activated, or call them through `.venv/bin/...`. To
get them on your `PATH` without activating the venv, symlink the venv's console
scripts into a directory already on `PATH` (e.g. `~/.local/bin`):

```
for s in "$PWD"/.venv/bin/hugin-meet-*; do
    ln -sf "$s" ~/.local/bin/"$(basename "$s")"
done
```

The console scripts carry an absolute shebang to the venv's Python, so the
symlinks resolve to the right interpreter from any directory. Re-run this after
recreating the venv. (Do **not** `pip install -e .` from a system/user Python to
get these on `PATH` — that regenerates the wrappers with a `#!/usr/bin/python3`
shebang that can't see the venv's CUDA stack.)

System dependencies (not pip-installable):

- `ffmpeg` (audio conversion)
- `codex`, `claude`, or `agy` CLI if you want to summarize via remote models
- `gws` (Google Workspace CLI) if you want calendar matching

For the optional GNOME tray widget, install `frontends/gnome/` separately
and run the bundled installer:

```
python -m pip install -e frontends/gnome
# plus Gtk system libs: apt install python3-gi gir1.2-ayatanaappindicator3-0.1
.venv/bin/hugin-meet-install-gnome-tray --venv "$PWD/.venv"            # writes the .desktop files
.venv/bin/hugin-meet-install-gnome-tray --venv "$PWD/.venv" --dry-run  # preview without writing
```

If you move or recreate the virtualenv, rerun the installer with `--force` so
GNOME launches the same environment later:

```
.venv/bin/hugin-meet-install-gnome-tray --venv "$PWD/.venv" --force
```

When launching the tray manually from a repo checkout, call the bundled launcher
script rather than `hugin-meet-recorder` directly. The launcher will infer
`$PWD/.venv` from the checkout path; set `HUGIN_MEETINGS_VENV=/path/to/venv`
only if you want to override that.

The installer drops `hugin-recorder.desktop` into both
`~/.local/share/applications/` (app menu) and `~/.config/autostart/`
(launches on login), with the absolute path to the bundled
`launcher.sh` substituted in. Existing files are left alone unless you
pass `--force`; pass `--no-autostart` if you only want the menu entry.

## Recording devices

When recording starts, Hugin uses the audio devices that are currently set as
the system defaults. The microphone track records from the default input
source; the system-audio track records from the monitor of the default output
sink. This applies whether recording is started from a frontend or a CLI entry
point, so set the correct microphone and speaker/output device before starting
the recording.

## Configure

Run `hugin-init` (shipped with the `hugin` shared library) to scaffold
`~/.config/hugin/hugin.yaml` and a vault layout. Then copy
`config.example.yaml` into `~/.config/hugin/meetings.yaml` for the
meetings-specific bits.

- `hugin.yaml` — shared across all hugin-* tools (language, vault, gws, ...)
- `meetings.yaml` — meetings-specific (transcripts dir, LLM, matcher, ...)

`meetings.yaml` overrides `hugin.yaml`. Every field has a sensible default
(English, generic summary prompt, matcher disabled), so a minimal config is fine.

Override the config dir with `HUGIN_CONFIG_DIR=/path`.

Transcripts and summaries are written to `meetings.transcripts_dir` and
`meetings.summaries_dir`, usually inside your vault. Runtime state is kept under
`meetings.state_dir` (default: `~/.hugin_audio`), including raw Opus audio in
`raw/`, cached WAV files, transcript JSON, speaker embeddings, and downloaded
models. These files are kept so recordings can be rescanned, reprocessed, or
used for later speaker enrollment. To purge a single meeting entry from the TUI,
select it and press `d`; this deletes the raw audio, transcript, summary, cached
WAVs, and the context and customer-state JSON for that meeting, while leaving
project/customer notes untouched.

The summary and project/customer matcher prompts are ordinary Markdown
templates under `src/hugin_meetings/prompts/`. Set
`meetings.summarize_prompt_path` or `meetings.project_matcher.prompt_path` to
use your own versions. Matcher templates can use `{{candidate_context}}`,
`{{calendar_lines}}`, `{{summary_body}}`, `{{internal_rules}}`, and
`{{evidence_rules}}` (filled in only when the match is made from the calendar
alone, before there is any meeting text).

Remote and local command models use `meetings.llm.provider`, which can be
`codex`, `claude`, `agy`, or `local`. The default is `codex`. Claude Code
runs from a clean working directory so repo-local `CLAUDE.md` files are not
discovered while normal Claude Code login still works. Antigravity also runs
from a clean working directory, in plan mode with its CLI sandbox enabled.

Set `summary_model` or `project_matcher.model` to `default` to let the provider
CLI choose its configured model. Summaries default to `summary_effort: high`;
project matching defaults to `project_matcher.effort: low`. Effort is applied
for Codex, Claude, and Antigravity.

For local one-shot inference, use `meetings.llm.provider: local` and set
`meetings.llm.local_command`. The prompt is written to stdin and stdout is used
as the model response, which works well for llama.cpp-style commands that should
exit and release GPU memory after each request.

## CLI

After activating the venv or installing with `python -m pip install -e .`:

| Command | What |
|---------|------|
| `hugin-meet-record` | Record mic + system audio until Ctrl-C (headless; no frontend needed) |
| `hugin-meet-context [session-id] [-n N]` | Guess language, calendar event and customer for a session |
| `hugin-meet-langid [session-id] [-n N]` | Just the language probe, for checking a batch of sessions |
| `hugin-meet-transcribe [session-id\|file]` | Transcribe + diarize a recording session (requires a context) |
| `hugin-meet-summarize [transcript]` | Summarize a transcript |
| `hugin-meet-match-calendar [transcript]` | Attach calendar metadata (`--rematch` to query Google again) |
| `hugin-meet-link [session-id] [--all]` | Publish a summary into its verified customer note |
| `hugin-meet-slack-post [summary] [--channel #name]` | Post summary to Slack |
| `hugin-meet-enroll` | Interactively enroll a new speaker |
| `hugin-meet-tui` | Interactive curses TUI — drives the whole pipeline |

Calendar matching searches calendars you own by default, which keeps shared or
subscribed calendars from being picked accidentally. Use
`hugin-meet-match-calendar --calendar <id>` for one specific calendar, or
`--include-shared-calendars` if shared calendars should be considered too.

## Slack integration

`hugin-meet-slack-post` posts a meeting summary to a Slack channel. It
requires a Slack bot token and a `slack_channel` entry in the linked
project file's frontmatter.

**One-time bot setup:**

1. Go to `api.slack.com/apps` → **Create New App** → **From scratch**
2. *OAuth & Permissions* → Bot Token Scopes → add `chat:write`
3. **Install to Workspace** → copy the `xoxb-…` Bot Token
4. `export SLACK_BOT_TOKEN=xoxb-…` (add to your shell profile)

**Per-channel setup:** in each Slack channel you want to post to, run
`/invite @YourBotName`.

**Per-project config:** add a `slack_channel` key to the YAML frontmatter
at the top of the project note:

```yaml
---
slack_channel: "#proj-acme"
---

# Acme Corp
```

Once configured, `hugin-meet-slack-post` (or `s` in the TUI after a
project is linked) will post the meeting title and abstract as a Block Kit
message, with the full summary (minus any personal-notes section) as a
plain-text thread reply. Pass `--channel` to override the frontmatter
value for a one-off post.

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
- Read pipeline state via `hugin_meetings.pipeline.scan_recordings()` and
  `MeetingStatus.needs_pipeline` / `.needs_verification`
- Hand off processing by launching `hugin-meet-tui`. Frontends do **not** call
  the stage CLIs themselves — those have ordering requirements, and a session
  has to be verified by a person before it is processed. Stopping a recording
  already kicks off the context guess on its own.

## Layout

```
src/hugin_meetings/
  config.py               Config loader (hugin.yaml + meetings.yaml)
  pipeline.py             Metadata, customer notes, shared pipeline helpers
  recording.py            Recording primitives used by frontends
  context.py              First stage: language + calendar + customer, and verification
  langid.py               Language identification from the audio
  transcribe.py           Whisper + diarization
  transcribe_part.py      Per-part worker (spawned as subprocess)
  summarize.py            LLM summarization
  enroll.py               Speaker enrollment
  calendar_match.py       Google Calendar matching
  link.py                 Last stage: publishes a summary into its customer note
  tui.py                  Interactive TUI
  slack_post.py           Slack posting
  prompts/                Summary and matcher prompt templates

frontends/
  gnome/                  GNOME tray widget (separate install target)
```

## Status

Early. Config boundary stable, internals still in flux. MIT licensed.
