"""Run prompts through remote coding-agent CLIs."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .config import DEFAULT_REMOTE_MODEL, LLMConfig


def _uses_default_model(model: str | None) -> bool:
    return not model or model == DEFAULT_REMOTE_MODEL


def _clean_cwd(cfg: LLMConfig) -> Path:
    cfg.clean_cwd.mkdir(parents=True, exist_ok=True)
    return cfg.clean_cwd


def _run_checked(cmd: list[str], prompt: str, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=prompt,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _raise_for_failure(provider: str, result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout).strip()
    raise RuntimeError(detail or f"{provider} prompt failed")


def _run_codex(cfg: LLMConfig, model: str, prompt: str, effort: str | None, timeout: int) -> str:
    cwd = _clean_cwd(cfg)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as out:
        out_path = Path(out.name)

    try:
        cmd = [
            cfg.codex_bin,
            "exec",
            "-C",
            str(cwd),
            "--skip-git-repo-check",
            "--ephemeral",
        ]
        if not _uses_default_model(model):
            cmd.extend(["-m", model])
        if effort:
            cmd.extend(["-c", f"model_reasoning_effort={effort}"])
        cmd.extend([*cfg.codex_args, "-o", str(out_path), "-"])
        result = _run_checked(cmd, prompt, cwd, timeout)
        _raise_for_failure("codex", result)
        return out_path.read_text(encoding="utf-8").strip()
    finally:
        out_path.unlink(missing_ok=True)


def _run_claude(cfg: LLMConfig, model: str, prompt: str, effort: str | None, timeout: int) -> str:
    cwd = _clean_cwd(cfg)
    cmd = [
        cfg.claude_bin,
        *cfg.claude_args,
        "--print",
        "--output-format",
        "text",
        "--no-session-persistence",
        "--tools",
        "",
    ]
    if not _uses_default_model(model):
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    result = _run_checked(cmd, prompt, cwd, timeout)
    _raise_for_failure("claude", result)
    return result.stdout.strip()


def _prepare_gemini_cwd(cfg: LLMConfig) -> Path:
    cwd = _clean_cwd(cfg)
    if cfg.gemini_disable_context:
        settings_dir = cwd / ".gemini"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "context": {
                "fileName": cfg.gemini_context_file_name,
                "includeDirectoryTree": False,
                "discoveryMaxDirs": 0,
            },
            "ui": {
                "hideBanner": True,
                "hideTips": True,
            },
        }
        (settings_dir / "settings.json").write_text(
            json.dumps(settings, indent=2) + "\n",
            encoding="utf-8",
        )
    return cwd


def _run_gemini(cfg: LLMConfig, model: str, prompt: str, effort: str | None, timeout: int) -> str:
    cwd = _prepare_gemini_cwd(cfg)
    cmd = [
        cfg.gemini_bin,
        "--prompt",
        "",
        "--output-format",
        "text",
        "--raw-output",
        "--accept-raw-output-risk",
        *cfg.gemini_args,
    ]
    if not _uses_default_model(model):
        cmd[1:1] = ["--model", model]
    result = _run_checked(cmd, prompt, cwd, timeout)
    _raise_for_failure("gemini", result)
    return result.stdout.strip()


def _run_local_command(cfg: LLMConfig, model: str, prompt: str, effort: str | None, timeout: int) -> str:
    if not cfg.local_command:
        raise RuntimeError("llm.local_command must be configured when llm.provider is local")

    cwd = _clean_cwd(cfg)
    model_value = "" if _uses_default_model(model) else model
    effort_value = effort or ""
    cmd = [
        part.replace("{model}", model_value).replace("{effort}", effort_value)
        for part in cfg.local_command
    ]
    result = _run_checked(cmd, prompt, cwd, timeout)
    _raise_for_failure("local", result)
    return result.stdout.strip()


def run_prompt(
    cfg: LLMConfig,
    model: str,
    prompt: str,
    effort: str | None = None,
    timeout: int = 300,
) -> str:
    if cfg.provider == "codex":
        return _run_codex(cfg, model, prompt, effort, timeout)
    if cfg.provider == "claude":
        return _run_claude(cfg, model, prompt, effort, timeout)
    if cfg.provider == "gemini":
        return _run_gemini(cfg, model, prompt, effort, timeout)
    if cfg.provider == "local":
        return _run_local_command(cfg, model, prompt, effort, timeout)
    raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
