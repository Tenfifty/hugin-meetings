"""Configuration loading for Hugin Meetings.

Reads ~/.config/hugin/hugin.yaml + ~/.config/hugin/meetings.yaml via
:func:`hugin.config.load_tool`. Override the config dir with
``HUGIN_CONFIG_DIR``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from hugin.config import SharedConfig, load_tool
from hugin.llm import DEFAULT_REMOTE_MODEL, LLMConfig

DEFAULT_SUMMARY_EFFORT = "high"
DEFAULT_PROJECT_MATCHER_EFFORT = "low"


def _opt_path(value: Any) -> Path | None:
    return Path(value).expanduser() if value else None


@dataclass
class ProjectMatcherConfig:
    """Matches meetings to project/customer notes in a directory.

    ``internal_project`` is the name of the note representing your own
    organization (given priority during matching). Originally "Tenfifty"
    for the author; set to whatever makes sense for you, or leave empty.
    """

    projects_dir: Path | None = None
    internal_project: str = ""
    model: str = DEFAULT_REMOTE_MODEL
    effort: str = DEFAULT_PROJECT_MATCHER_EFFORT
    prompt_path: Path | None = None
    json_system_prompt: str = "Return only valid JSON."
    inactive_dir_names: list[str] = field(default_factory=lambda: ["inactive", "inaktiva"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectMatcherConfig":
        return cls(
            projects_dir=_opt_path(data.get("projects_dir")),
            internal_project=data.get("internal_project", ""),
            model=data.get("model", DEFAULT_REMOTE_MODEL),
            effort=data.get("effort", DEFAULT_PROJECT_MATCHER_EFFORT),
            prompt_path=_opt_path(data.get("prompt_path")),
            json_system_prompt=data.get("json_system_prompt", "Return only valid JSON."),
            inactive_dir_names=data.get("inactive_dir_names", ["inactive", "inaktiva"]),
        )


@dataclass
class MeetingsConfig(SharedConfig):
    # Output (goes into your vault / knowledge base)
    transcripts_dir: Path = field(default_factory=lambda: Path.home() / "hugin_meetings" / "transcripts")
    summaries_dir: Path = field(default_factory=lambda: Path.home() / "hugin_meetings" / "summaries")

    # State (caches, raw audio, models, speaker embeddings)
    state_dir: Path = field(default_factory=lambda: Path.home() / ".hugin_audio")

    # Transcription
    whisper_model: str = "large-v3"
    compute_type: str = "float16"

    # Local llama.cpp summarization tuning. ``summarize_n_gpu_layers``, if set,
    # overrides the auto pick (-1 = all on GPU). Otherwise models larger than
    # ``summarize_hybrid_threshold_gb`` load in hybrid mode with
    # ``summarize_hybrid_n_gpu_layers`` layers on the GPU.
    summarize_n_gpu_layers: int | None = None
    summarize_hybrid_threshold_gb: float = 10.0
    summarize_hybrid_n_gpu_layers: int = 10

    # Project/customer matching
    project_matcher: ProjectMatcherConfig = field(default_factory=ProjectMatcherConfig)

    # Remote LLM provider used for non-local models.
    llm: LLMConfig = field(default_factory=LLMConfig)
    summary_model: str = DEFAULT_REMOTE_MODEL
    summary_effort: str = DEFAULT_SUMMARY_EFFORT

    # Summary formatting — what the summarizer produces. Language-specific.
    summarize_prompt_path: Path | None = None
    # H2 heading marking the start of the summary
    # (e.g. "## Meeting Summary" or "## Mötessammanfattning").
    summary_header: str = "## Meeting Summary"
    # Optional H3 section carved out of the summary for personal follow-ups
    # (e.g. "### For Me" or "### För David"). Empty disables extraction.
    personal_section_header: str = ""

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

    @classmethod
    def from_merged(cls, merged: dict[str, Any]) -> "MeetingsConfig":
        meetings = merged.get("meetings", {}) if isinstance(merged.get("meetings"), dict) else {}
        defaults = cls()

        def _meet_path(key: str, default: Path | None) -> Path | None:
            return _opt_path(meetings.get(key)) or default

        return cls(
            **SharedConfig.fields_from_merged(merged),
            transcripts_dir=_meet_path("transcripts_dir", defaults.transcripts_dir) or defaults.transcripts_dir,
            summaries_dir=_meet_path("summaries_dir", defaults.summaries_dir) or defaults.summaries_dir,
            state_dir=_meet_path("state_dir", defaults.state_dir) or defaults.state_dir,
            whisper_model=meetings.get("whisper_model", "large-v3"),
            compute_type=meetings.get("compute_type", "float16"),
            summarize_n_gpu_layers=meetings.get("summarize_n_gpu_layers"),
            summarize_hybrid_threshold_gb=float(meetings.get("summarize_hybrid_threshold_gb", 10.0)),
            summarize_hybrid_n_gpu_layers=int(meetings.get("summarize_hybrid_n_gpu_layers", 10)),
            project_matcher=ProjectMatcherConfig.from_dict(meetings.get("project_matcher", {})),
            llm=LLMConfig.from_dict(meetings.get("llm", {})),
            summary_model=meetings.get("summary_model", DEFAULT_REMOTE_MODEL),
            summary_effort=meetings.get("summary_effort", DEFAULT_SUMMARY_EFFORT),
            summarize_prompt_path=_meet_path("summarize_prompt_path", None),
            summary_header=meetings.get("summary_header", "## Meeting Summary"),
            personal_section_header=meetings.get("personal_section_header", ""),
        )


@lru_cache(maxsize=1)
def load_config() -> MeetingsConfig:
    return load_tool("meetings", MeetingsConfig.from_merged)


def reset_config_cache() -> None:
    """For tests."""
    load_config.cache_clear()
