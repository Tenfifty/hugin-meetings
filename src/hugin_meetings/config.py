"""Configuration loading for Hugin Meetings.

Reads two YAML files and merges them (meetings.yaml overrides hugin.yaml):
- ~/.config/hugin/hugin.yaml   -- shared across all hugin-* tools
- ~/.config/hugin/meetings.yaml -- meetings-specific

Environment variable HUGIN_CONFIG_DIR overrides the config directory.
Individual values can also be overridden by env vars prefixed with HUGIN_MEET_.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _config_dir() -> Path:
    override = os.environ.get("HUGIN_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "hugin"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class ProjectMatcherConfig:
    """Matches meetings to project/customer notes in a directory.

    `internal_project` is the name of the note representing your own
    organization (given priority during matching). Originally "Tenfifty"
    for the author; set to whatever makes sense for you, or leave empty.
    """

    projects_dir: Path | None = None
    internal_project: str = ""
    model: str = "gpt-5.4-mini"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectMatcherConfig":
        projects_dir = data.get("projects_dir")
        return cls(
            projects_dir=Path(projects_dir).expanduser() if projects_dir else None,
            internal_project=data.get("internal_project", ""),
            model=data.get("model", "gpt-5.4-mini"),
        )


@dataclass
class MeetingsConfig:
    # Shared
    language: str = "en"
    user_name: str = ""
    vault_path: Path | None = None

    # Output (goes into your vault / knowledge base)
    transcripts_dir: Path = field(default_factory=lambda: Path.home() / "hugin_meetings" / "transcripts")
    summaries_dir: Path = field(default_factory=lambda: Path.home() / "hugin_meetings" / "summaries")

    # State (caches, raw audio, models, speaker embeddings)
    state_dir: Path = field(default_factory=lambda: Path.home() / ".hugin_audio")

    # Transcription
    whisper_model: str = "large-v3"
    compute_type: str = "float16"

    # Calendar (shared with other hugin-* tools; may live at top level)
    gws_bin: str = "gws"
    gws_config_dir: Path | None = None

    # Daily journal file (shared; may live at top level)
    journal_path: Path | None = None

    # Project/customer matching
    project_matcher: ProjectMatcherConfig = field(default_factory=ProjectMatcherConfig)

    # Stopwords for fuzzy matching (lang-specific)
    stopwords: list[str] = field(default_factory=list)

    # Summary formatting — what the summarizer produces. Language-specific.
    # summary_header is the H2 heading that marks the start of the summary
    # (e.g. "## Meeting Summary" or "## Mötessammanfattning").
    summary_header: str = "## Meeting Summary"
    # Optional H3 section carved out of the summary for personal follow-ups
    # (e.g. "### For Me" or "### För David"). Empty disables extraction.
    personal_section_header: str = ""

    # Raw merged dict for anything not explicitly modeled
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_audio_dir(self) -> Path:
        return self.state_dir / "raw"

    @property
    def wav_cache_dir(self) -> Path:
        return self.state_dir / "cache" / "wav"

    @property
    def speakers_dir(self) -> Path:
        return self.state_dir / "speakers"

    @property
    def models_dir(self) -> Path:
        return self.state_dir / "models"

    @property
    def transcript_json_dir(self) -> Path:
        return self.state_dir / "transcripts"

    @property
    def recorder_state_dir(self) -> Path:
        return self.state_dir / "state"


def _build(merged: dict[str, Any]) -> MeetingsConfig:
    merged = _expand(merged)
    meetings = merged.get("meetings", {}) if "meetings" in merged else merged

    def _path(key: str, default: Path | None = None) -> Path | None:
        value = meetings.get(key)
        if value:
            return Path(value).expanduser()
        return default

    cfg = MeetingsConfig(
        language=merged.get("language", "en"),
        user_name=merged.get("user_name", ""),
        vault_path=_path("vault_path") or (Path(merged["vault_path"]).expanduser() if merged.get("vault_path") else None),
        transcripts_dir=_path("transcripts_dir") or MeetingsConfig().transcripts_dir,
        summaries_dir=_path("summaries_dir") or MeetingsConfig().summaries_dir,
        state_dir=_path("state_dir") or MeetingsConfig().state_dir,
        whisper_model=meetings.get("whisper_model", "large-v3"),
        compute_type=meetings.get("compute_type", "float16"),
        gws_bin=meetings.get("gws_bin", merged.get("gws_bin", "gws")),
        gws_config_dir=_path("gws_config_dir") or (
            Path(merged["gws_config_dir"]).expanduser()
            if merged.get("gws_config_dir") else None
        ),
        journal_path=_path("journal_path") or (
            Path(merged["journal_path"]).expanduser()
            if merged.get("journal_path") else None
        ),
        project_matcher=ProjectMatcherConfig.from_dict(meetings.get("project_matcher", {})),
        stopwords=meetings.get("stopwords", []),
        summary_header=meetings.get("summary_header", "## Meeting Summary"),
        personal_section_header=meetings.get("personal_section_header", ""),
        raw=merged,
    )
    return cfg


@lru_cache(maxsize=1)
def load_config() -> MeetingsConfig:
    cfg_dir = _config_dir()
    shared = _load_yaml(cfg_dir / "hugin.yaml")
    meetings = _load_yaml(cfg_dir / "meetings.yaml")
    merged = _deep_merge(shared, meetings)
    return _build(merged)


def reset_config_cache() -> None:
    """For tests."""
    load_config.cache_clear()
