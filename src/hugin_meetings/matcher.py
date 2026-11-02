"""The LLM customer matcher: which project note a meeting belongs to.

Split out of pipeline (2026-11-02) so that scanning sessions, which the GNOME
tray does every few seconds, does not import the LLM runner and the prompt
resolver. Only the context stage guesses a customer, so only it imports this.
"""

from __future__ import annotations

import json
from pathlib import Path

from hugin.llm import run_prompt
from hugin.prompts import resolve_prompt

from .config import load_config
from .pipeline import (
    DEFAULT_CUSTOMER_MODEL,
    CustomerNote,
    CustomerState,
    _is_folder_name,
    list_customer_notes,
    resolve_customer_path,
)

INTERNAL_PROJECT = load_config().project_matcher.internal_project
DEFAULT_CUSTOMER_EFFORT = load_config().project_matcher.effort
MAX_MEETING_TEXT = 12000
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _relative_or_absolute(path: Path) -> str:
    """Return path relative to vault if possible, else absolute."""
    if load_config().vault_path:
        try:
            return str(path.relative_to(load_config().vault_path))
        except ValueError:
            pass
    return str(path)


def _load_customer_prompt_template() -> str:
    path = resolve_prompt(
        base="project_matcher",
        language=load_config().language,
        explicit=load_config().project_matcher.prompt_path,
        package_dir=_PROMPTS_DIR,
    )
    return path.read_text(encoding="utf-8")


def _render_customer_prompt_template(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def build_customer_prompt_from(
    calendar_fields: dict[str, str], meeting_text: str
) -> tuple[str, list[CustomerNote]]:
    """Render the matcher prompt from whatever the caller knows about a meeting.

    Before transcription that is the calendar event alone; afterwards the
    summary body carries most of the signal. Same template either way, so the
    two stages cannot drift apart.
    """
    has_text = bool(meeting_text.strip())
    summary_body = (
        meeting_text.strip()[:MAX_MEETING_TEXT]
        if has_text
        else "(not transcribed yet — judge from the calendar event above)"
    )
    # Without this the model reads "Meeting text: (not transcribed yet)" as
    # missing evidence and abstains, however clear the event is.
    evidence_rules = "" if has_text else (
        "- There is no meeting text yet. The calendar event is the complete evidence base: "
        "judge from its title, description, organizer and the attendees' mail domains, and do "
        "not abstain merely because no summary exists.\n"
        "- An external organization named in the title, or owning an attendee's mail domain, "
        "is concrete evidence. So is a customer named in the title of an otherwise internal meeting.\n"
    )

    calendar_lines = "\n".join(
        f"- {key}: {value}" for key, value in calendar_fields.items()
    ) or "- (no calendar metadata)"

    all_notes = list_customer_notes()

    candidate_context = "\n\n".join(
        [
            f"Customer: {note.name}\nPath: {_relative_or_absolute(note.path)}\nNotes: {note.preview}"
            for note in all_notes
        ]
    )

    internal_name = INTERNAL_PROJECT or "(internal)"
    internal_rules = (
        (
            f"- A meeting can still belong to a project even if all attendees are internal {internal_name} people.\n"
            f'- Choose "{internal_name}" only for genuinely internal meetings with no clear project/customer focus, such as admin, staffing, company process, internal tooling, internal strategy without a specific project, or similar operational topics.\n'
            f'- If a known project is discussed concretely, prefer that project over "{internal_name}".\n'
        )
        if INTERNAL_PROJECT
        else ""
    )

    prompt = _render_customer_prompt_template(
        _load_customer_prompt_template(),
        {
            "internal_rules": internal_rules,
            "evidence_rules": evidence_rules,
            "candidate_context": candidate_context,
            "calendar_lines": calendar_lines,
            "summary_body": summary_body,
        },
    )

    return prompt, all_notes


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No JSON object found in model output")


def run_remote_json_prompt(model: str, prompt: str, effort: str | None = None) -> dict:
    return extract_json_object(run_prompt(load_config().llm, model, prompt, effort=effort))


def suggest_customer_from_calendar(
    calendar_fields: dict[str, str],
    model: str = DEFAULT_CUSTOMER_MODEL,
    effort: str = DEFAULT_CUSTOMER_EFFORT,
) -> CustomerState:
    """Match a meeting from its calendar event alone, before it is transcribed.

    The event carries the mail domains, the title and the description, which is
    usually enough to name the customer — and when it is not, the guess is
    unverified anyway and the operator gets to correct it.
    """
    prompt, candidates = build_customer_prompt_from(calendar_fields, "")
    return decide_customer(prompt, candidates, model, effort, source="calendar")


def decide_customer(
    prompt: str,
    candidates: list[CustomerNote],
    model: str = DEFAULT_CUSTOMER_MODEL,
    effort: str = DEFAULT_CUSTOMER_EFFORT,
    *,
    source: str = "auto",
) -> CustomerState:
    """An unverified guess: what the model answered, checked against the notes."""
    payload = run_remote_json_prompt(model, prompt, effort=effort)

    action = str(payload.get("action", "no_match")).strip()
    if action not in {"link_existing", "suggest_new", "no_match"}:
        action = "no_match"

    confidence = str(payload.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    customer_name = payload.get("customer_name") or None
    suggested_name = payload.get("suggested_name") or None
    rationale = str(payload.get("rationale", "")).strip()

    # Folder names are never guesses: a material folder named after its note
    # resolves to the note, and any other folder is not a project at all.
    if _is_folder_name(customer_name):
        rationale = f"{rationale} (Ignored {customer_name!r}: a folder, not a project.)".strip()
        customer_name = None
    if _is_folder_name(suggested_name):
        rationale = f"{rationale} (Ignored suggestion {suggested_name!r}: a folder, not a project.)".strip()
        suggested_name = None

    customer_path = resolve_customer_path(customer_name, candidates)
    # A "new" project that already has a note under that name is a link, not a second note.
    existing = resolve_customer_path(suggested_name, candidates) if action == "suggest_new" else None
    if existing:
        action, customer_name, customer_path, suggested_name = "link_existing", existing.stem, existing, None
    if action == "link_existing" and not customer_path:
        action = "suggest_new" if suggested_name else "no_match"
    if action == "suggest_new" and not suggested_name:
        action = "no_match"

    return CustomerState(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        rationale=rationale or "No clear rationale returned.",
        model=model,
        customer_name=customer_name,
        customer_path=customer_path,
        suggested_name=suggested_name,
        source=source,
    )
