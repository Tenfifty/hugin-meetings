#!/usr/bin/env python3
"""Post a meeting summary to a Slack channel."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .cli_utils import resolve_summary_md
from .config import load_config
from .pipeline import (
    PERSONAL_SECTION_HEADER,
    SUMMARY_DIR,
    TRANSCRIPT_DIR,
    _strip_customer_header_link,
    _strip_transcript_link,
    extract_timestamp,
    load_customer_state,
    parse_calendar_metadata,
    read_project_frontmatter,
    year_subdir,
)

_SLACK_API = "https://slack.com/api/"


def _token() -> str:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        print("SLACK_BOT_TOKEN not set.", file=sys.stderr)
        sys.exit(1)
    return token


def _api(token: str, method: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = Request(
        _SLACK_API + method,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except URLError as exc:
        print(f"Slack request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    if not result.get("ok"):
        print(f"Slack error: {result.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return result


def _table_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.match(r"^:?-+:?$", c) for c in cells if c)


def _render_table(table_lines: list[str]) -> str:
    rows = [_table_cells(l) for l in table_lines]
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    sep_indices = {i for i, r in enumerate(rows) if _is_separator(r)}
    widths = [0] * n_cols
    for i, row in enumerate(rows):
        if i not in sep_indices:
            for j, cell in enumerate(row):
                widths[j] = max(widths[j], len(cell))
    rendered = []
    for i, row in enumerate(rows):
        if i in sep_indices:
            cells = ["-" * widths[j] for j in range(n_cols)]
        else:
            cells = [row[j].ljust(widths[j]) for j in range(n_cols)]
        rendered.append("| " + " | ".join(cells) + " |")
    return "```\n" + "\n".join(rendered) + "\n```"


def _convert_tables(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            table: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i])
                i += 1
            result.append(_render_table(table))
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def _md_to_mrkdwn(text: str) -> str:
    # Tables first (content stays literal inside code blocks)
    text = _convert_tables(text)
    # Bold: consume ** before italic pass to prevent double-conversion
    text = re.sub(r"\*\*(.+?)\*\*", "\x00" + r"\1" + "\x00", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", "\x00" + r"\1" + "\x00", text, flags=re.DOTALL)
    # Links: [label](url) -> <url|label>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    # Italic (safe now ** is consumed)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", text)
    # Restore bold sentinel
    text = text.replace("\x00", "*")
    # Headings -> bold line
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    return text


def _summary_body(summary_path: Path) -> str:
    """Cleaned summary text: no header line, no transcript/customer links."""
    text = _strip_transcript_link(_strip_customer_header_link(summary_path.read_text())).strip()
    lines = text.splitlines()
    if lines and lines[0].startswith(load_config().summary_header):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _strip_personal(body: str) -> str:
    if not PERSONAL_SECTION_HEADER:
        return body
    pattern = re.compile(
        rf"^{re.escape(PERSONAL_SECTION_HEADER)}\s*$.*?(?=^### |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    return re.sub(r"\n{3,}", "\n\n", pattern.sub("", body)).strip()


def _lead(body: str) -> str:
    """Text before the first ### subsection."""
    lines: list[str] = []
    for line in body.splitlines():
        if line.strip().startswith("### "):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _get_slack_channel(summary_path: Path) -> str | None:
    ts = extract_timestamp(summary_path.name)
    if not ts:
        return None
    state = load_customer_state(ts, summary_path)
    if not state or not state.customer_path or not state.customer_path.exists():
        return None
    return read_project_frontmatter(state.customer_path).get("slack_channel")


def _abstract_blocks(
    title: str,
    lead_text: str,
    calendar_fields: dict[str, str],
) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": title[:150], "emoji": False},
        }
    ]
    if lead_text:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": _md_to_mrkdwn(lead_text)[:3000]},
        })
    if date := calendar_fields.get("Date") or calendar_fields.get("Start"):
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"*Date:* {date}"}],
        })
    blocks.append({"type": "divider"})
    return blocks


def post_summary(summary_path: Path, channel: str | None = None) -> None:
    if not summary_path.exists():
        print(f"Summary not found: {summary_path}", file=sys.stderr)
        sys.exit(1)

    resolved_channel = channel or _get_slack_channel(summary_path)
    if not resolved_channel:
        print(
            "No Slack channel found. Add 'slack_channel: \"#channel\"' to the project file frontmatter.",
            file=sys.stderr,
        )
        sys.exit(1)

    token = _token()

    ts = extract_timestamp(summary_path.name) or ""

    # Calendar metadata lives in the transcript, not the summary
    transcript_path: Path | None = None
    if ts:
        transcript_path = TRANSCRIPT_DIR / year_subdir(ts) / f"transcript-{ts}.md"
    calendar_fields = parse_calendar_metadata(transcript_path)

    event = calendar_fields.get("Event") or ""
    title = event if event and event != "-" else f"Meeting {ts}"

    body = _summary_body(summary_path)
    lead_text = _lead(body)
    thread_body = _strip_personal(body)

    abstract = _abstract_blocks(title, lead_text, calendar_fields)
    result = _api(token, "chat.postMessage", {
        "channel": resolved_channel,
        "blocks": abstract,
        "text": f"Meeting: {title}",
    })
    thread_ts = result["ts"]
    print(f"Posted to {resolved_channel} (ts={thread_ts})")

    if thread_body:
        _api(token, "chat.postMessage", {
            "channel": resolved_channel,
            "thread_ts": thread_ts,
            "text": _md_to_mrkdwn(thread_body),
        })
        print("Posted full summary as thread reply.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post a meeting summary to Slack")
    parser.add_argument("summary", nargs="?", help="Summary .md or timestamp (default: latest)")
    parser.add_argument("--channel", help="Override Slack channel (default: from project frontmatter)")
    args = parser.parse_args()

    summary_path = resolve_summary_md(SUMMARY_DIR, args.summary)
    post_summary(summary_path, channel=args.channel)


if __name__ == "__main__":
    main()
