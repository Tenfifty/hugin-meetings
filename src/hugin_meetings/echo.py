"""Cancel the far end out of the mic track before it is transcribed.

On a call through the laptop's speakers the mic hears the remote side too,
and Whisper transcribes that echo a little differently from the clean sys
track, so the text dedup in ``transcribe._dedup_echo`` misses most of it and
the diarizer invents speakers on the mic. On 2026-10-30
5692 of the mic transcript's 7023 words were echo of sys.

The sys track is the exact signal the speakers played, so it is a reference
for acoustic echo cancellation: WebRTC AEC3, through LiveKit's build of the
audio processing module. Measured on 2026-10-28 and 2026-10-30, it brings the
echo down to the mic's own noise floor, leaves David speaking alone within
0.6 dB, and loses a phrase now and then where he talks over the remote side
(its suppressor takes about 12 dB off him there).

The two tracks come from two ffmpeg processes, so the echo trails its
reference by the recorders' start gap plus output latency: 212 and 361 ms on
those two days, constant within a session to 0.3 ms. It is measured per part
with GCC-PHAT and the reference shifted to lead its echo by ``LEAD_MS``. No
confident peak means no echo (headphones): the mic is transcribed as it is.

The raw recordings are never changed; the cancelled mic is a temporary WAV.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
FRAME = SR // 100           # AEC3 takes 10 ms frames
LEAD_MS = 20.0              # how far the shifted reference leads its echo
MAX_DELAY_S = 1.0
WINDOW_S, HOP_S = 20.0, 60.0
MIN_PEAK = 8.0              # GCC-PHAT peak over its median: an echo path
MIN_ECHO_WINDOWS = 3
MIN_REF_RMS = 1e-3


@dataclass
class EchoReport:
    delay_ms: float
    echo_windows: int
    active_windows: int
    spread_ms: float

    def describe(self) -> str:
        return (f"echo delay {self.delay_ms:.1f} ms (spread {self.spread_ms:.1f} ms) "
                f"in {self.echo_windows}/{self.active_windows} windows")


def load_pcm(path: Path, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _gcc_phat(mic: np.ndarray, ref: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag in samples by which mic trails ref, and the peak over the median."""
    n = len(mic) + len(ref)
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(mic, nfft) * np.conj(np.fft.rfft(ref, nfft))
    spec /= np.abs(spec) + 1e-12
    cc = np.fft.irfft(spec, nfft)
    cc = np.abs(np.concatenate([cc[-max_lag:], cc[: max_lag + 1]]))
    i = int(np.argmax(cc))
    return i - max_lag, float(cc[i] / (np.median(cc) + 1e-12))


def measure_echo(mic: np.ndarray, ref: np.ndarray, sr: int = SR) -> EchoReport | None:
    """The echo delay over windows where sys plays, or None when there is no echo."""
    win, hop, max_lag = int(WINDOW_S * sr), int(HOP_S * sr), int(MAX_DELAY_S * sr)
    n = min(len(mic), len(ref))
    if n < win:  # a short part: one window over all of it
        win = n
    starts = range(0, max(n - win, 0) + 1, hop)
    delays, active = [], 0
    for start in starts:
        r = ref[start:start + win]
        if len(r) < sr or np.sqrt(np.mean(r ** 2)) < MIN_REF_RMS:
            continue
        active += 1
        lag, peak = _gcc_phat(mic[start:start + win], r, min(max_lag, len(r) - 1))
        if peak >= MIN_PEAK and lag >= 0:
            delays.append(lag / sr * 1000)
    if len(delays) < min(MIN_ECHO_WINDOWS, max(active, 1)):
        return None
    return EchoReport(
        delay_ms=float(np.median(delays)),
        echo_windows=len(delays),
        active_windows=active,
        spread_ms=float(np.percentile(delays, 90) - np.percentile(delays, 10)),
    )


def align_reference(ref: np.ndarray, length: int, delay_ms: float, sr: int = SR) -> np.ndarray:
    """Shift ref so it leads its echo in the mic by LEAD_MS, cut or padded to length."""
    shift = int(round((delay_ms - LEAD_MS) / 1000 * sr))
    if shift > 0:
        ref = np.concatenate([np.zeros(shift, np.float32), ref])
    else:
        ref = ref[-shift:]
    return np.pad(ref, (0, max(0, length - len(ref))))[:length].astype(np.float32)


def run_aec3(mic: np.ndarray, ref: np.ndarray, sr: int = SR) -> np.ndarray:
    """WebRTC AEC3 over the whole track, reference and mic frame by frame."""
    from livekit import rtc

    apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
    m16 = (np.clip(mic, -1, 1) * 32767).astype(np.int16)
    r16 = (np.clip(ref, -1, 1) * 32767).astype(np.int16)
    out = m16.copy()  # a trailing partial frame passes through untouched
    n = sr // 100
    for i in range(0, len(m16) - n + 1, n):
        apm.process_reverse_stream(rtc.AudioFrame(r16[i:i + n].tobytes(), sr, 1, n))
        frame = rtc.AudioFrame(m16[i:i + n].tobytes(), sr, 1, n)
        apm.process_stream(frame)
        out[i:i + n] = np.frombuffer(bytes(frame.data), dtype=np.int16)
    return out.astype(np.float32) / 32768.0


def cancel_echo(mic_path: Path, sys_path: Path, out_path: Path) -> EchoReport | None:
    """Write the mic with sys cancelled out to ``out_path`` (16 kHz WAV).

    Returns None, and writes nothing, when the mic holds no echo of sys.
    """
    import soundfile as sf

    mic, ref = load_pcm(mic_path), load_pcm(sys_path)
    report = measure_echo(mic, ref)
    if report is None:
        return None
    cleaned = run_aec3(mic, align_reference(ref, len(mic), report.delay_ms))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, cleaned, SR, subtype="PCM_16")
    return report
