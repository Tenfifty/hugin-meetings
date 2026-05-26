"""Audio input route discovery.

The current implementation targets Linux PipeWire/PulseAudio. Keeping it in
core makes it reusable by any Linux frontend while leaving room for other
platform-specific route providers later.
"""

from __future__ import annotations

import json
import logging
import subprocess

DEFAULT_PULSE_SOURCE = "default"
DEFAULT_MONITOR_SOURCE = "default.monitor"


def load_pipewire_nodes() -> list[dict] | None:
    try:
        result = subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=5, check=True
        )
        nodes = json.loads(result.stdout)
    except Exception:
        logging.exception("Failed to inspect PipeWire nodes")
        return None

    return nodes if isinstance(nodes, list) else None


def default_pulse_monitor_source() -> str:
    """Return the PulseAudio/PipeWire monitor for the current default sink."""
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        logging.info("Could not inspect PulseAudio default sink; using generic monitor fallback")
        return DEFAULT_MONITOR_SOURCE

    sink = result.stdout.strip()
    return f"{sink}.monitor" if sink else DEFAULT_MONITOR_SOURCE


def _metadata_name(value):
    if isinstance(value, dict):
        name = value.get("name")
        return name if isinstance(name, str) and name else None
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        name = parsed.get("name")
        return name if isinstance(name, str) and name else None
    return None


def _node_props(node):
    return node.get("info", {}).get("props") or node.get("props", {})


def _node_metadata(node):
    return node.get("info", {}).get("metadata") or node.get("metadata", [])


def _default_pipewire_node_name(nodes, media_class):
    default_key = f"default.audio.{media_class.removeprefix('Audio/').lower()}"
    configured_key = f"default.configured.audio.{media_class.removeprefix('Audio/').lower()}"
    candidates = {}

    for node in nodes:
        props = _node_props(node)
        if props.get("metadata.name") != "default":
            continue
        for item in _node_metadata(node):
            key = item.get("key")
            if key in {default_key, configured_key}:
                candidates[key] = _metadata_name(item.get("value"))

    for key in (default_key, configured_key):
        name = candidates.get(key)
        if name:
            return name
    return None


def _first_pipewire_node_name(nodes, media_class):
    for node in nodes:
        props = _node_props(node)
        if props.get("media.class") == media_class:
            name = props.get("node.name")
            if name:
                return name
    return None


def resolve_default_audio_source(nodes):
    source = _default_pipewire_node_name(nodes, "Audio/Source")
    return source or _first_pipewire_node_name(nodes, "Audio/Source") or DEFAULT_PULSE_SOURCE


def resolve_default_monitor_source(nodes):
    sink = _default_pipewire_node_name(nodes, "Audio/Sink")
    if sink:
        return f"{sink}.monitor"

    sink = _first_pipewire_node_name(nodes, "Audio/Sink")
    if sink:
        return f"{sink}.monitor"

    return DEFAULT_MONITOR_SOURCE


def get_default_audio_routes(log: bool = True) -> tuple[str, str]:
    """Get the current mic and monitor sources for ffmpeg's pulse inputs."""
    nodes = load_pipewire_nodes()
    if nodes is None:
        monitor_source = default_pulse_monitor_source()
        if log:
            logging.info(
                "Falling back to PulseAudio routes: mic=%s sys=%s",
                DEFAULT_PULSE_SOURCE,
                monitor_source,
            )
        return DEFAULT_PULSE_SOURCE, monitor_source

    mic_source = resolve_default_audio_source(nodes)
    monitor_source = resolve_default_monitor_source(nodes)
    if log:
        logging.info("Using audio routes: mic=%s sys=%s", mic_source, monitor_source)
    return mic_source, monitor_source


def get_default_audio_source() -> str:
    return get_default_audio_routes()[0]


def get_default_monitor_source() -> str:
    return get_default_audio_routes()[1]
