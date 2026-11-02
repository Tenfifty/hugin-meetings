import numpy as np
import pytest

from hugin_meetings import echo

SR = echo.SR


def _bursts(rng, seconds, on):
    """Noise that is on in the given (start, end) second ranges."""
    x = np.zeros(int(seconds * SR), np.float32)
    for a, b in on:
        x[int(a * SR):int(b * SR)] = rng.standard_normal(int((b - a) * SR)) * 0.1
    return x


def _db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


@pytest.fixture
def call():
    """A far end heard through a room 250 ms later, and a near end at other times."""
    rng = np.random.default_rng(1)
    far = _bursts(rng, 40, [(1, 12), (20, 30)])
    near = _bursts(rng, 40, [(13, 19), (32, 38)])
    room = rng.standard_normal(400) * np.exp(-np.arange(400) / 60)  # 25 ms decay
    room *= 0.3 / np.sqrt(np.sum(room ** 2))
    delay = int(0.250 * SR)
    heard = np.convolve(far, room)[: len(far)]
    heard = np.concatenate([np.zeros(delay, np.float32), heard])[: len(far)]
    mic = heard + near + rng.standard_normal(len(far)).astype(np.float32) * 1e-4
    return mic.astype(np.float32), far, near


def test_measure_echo_finds_the_delay(call):
    mic, far, _ = call
    report = echo.measure_echo(mic, far)
    assert report is not None
    assert report.delay_ms == pytest.approx(250, abs=2)


def test_no_echo_means_no_report():
    rng = np.random.default_rng(2)
    mic = _bursts(rng, 40, [(2, 30)])
    far = _bursts(rng, 40, [(1, 35)])
    assert echo.measure_echo(mic, far) is None


def test_silent_reference_means_no_report(call):
    mic, far, _ = call
    assert echo.measure_echo(mic, np.zeros_like(far)) is None


def test_align_reference_leads_the_echo_by_lead_ms():
    ref = np.arange(1, 10001, dtype=np.float32)
    out = echo.align_reference(ref, 10000, delay_ms=100)
    shift = int((100 - echo.LEAD_MS) / 1000 * SR)
    assert out[shift] == 1 and not out[:shift].any()
    assert len(out) == 10000
    early = echo.align_reference(ref, 10000, delay_ms=0)
    assert early[0] == ref[int(echo.LEAD_MS / 1000 * SR)]


def test_aec3_removes_echo_and_keeps_the_near_end(call):
    pytest.importorskip("livekit.rtc")
    mic, far, near = call
    report = echo.measure_echo(mic, far)
    out = echo.run_aec3(mic, echo.align_reference(far, len(mic), report.delay_ms))
    assert len(out) == len(mic)
    # Echo alone, after the filter has converged.
    s = slice(int(22 * SR), int(30 * SR))
    assert _db(out[s]) < _db(mic[s]) - 20
    # Near end alone.
    s = slice(int(33 * SR), int(38 * SR))
    assert _db(out[s]) == pytest.approx(_db(mic[s]), abs=1.5)
