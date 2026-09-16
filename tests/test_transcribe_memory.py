import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from hugin_meetings import transcribe as t


def test_gpu_asr_starts_with_batch_four(monkeypatch):
    calls = []

    def run(audio, batch_size, language):
        calls.append(batch_size)
        return {"segments": []}

    cuda = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(
        load_audio=lambda path: path,
        load_align_model=lambda **kwargs: (object(), {}),
        align=lambda *args, **kwargs: {"segments": []},
    ))
    t.transcribe(Path("mic"), SimpleNamespace(transcribe=run), "cuda", "sv")
    assert calls == [4]


@pytest.fixture
def runtime(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None, reset_peak_memory_stats=lambda: None)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, OutOfMemoryError=MemoryError))
    monkeypatch.setitem(sys.modules, 'whisperx', SimpleNamespace(load_audio=lambda path: path))
    monkeypatch.setattr(t, '_cuda_memory_stats', lambda: 'fake-memory')


def test_asr_reuses_reduced_batch_across_tracks(runtime):
    calls = []
    def run(audio, batch_size, language):
        calls.append((audio, batch_size))
        if batch_size > 2:
            raise RuntimeError('CUDA out of memory')
        return {'segments': []}
    state = {}
    for path in ('mic', 'sys'):
        t.transcribe(Path(path), SimpleNamespace(transcribe=run), 'cuda', 'sv', batch_state=state)
    assert calls == [('mic', 4), ('mic', 2), ('sys', 2)]


def test_cpu_oom_is_not_batch_retried(runtime):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs['batch_size'])
        raise RuntimeError('out of memory')
    with pytest.raises(RuntimeError):
        t.transcribe(Path('mic'), SimpleNamespace(transcribe=run), 'cpu', 'sv')
    assert calls == [8]


def test_stage_releases_failed_model_and_retains_completed_track(runtime):
    calls, references = [], []
    class Model:
        pass
    def load(device):
        if device == 'cpu':
            assert references[0]() is None
        model = Model()
        references.append(weakref.ref(model))
        return model
    def run(path, previous, model, device):
        calls.append((str(path), device))
        if str(path) == 'sys' and device == 'cuda':
            raise RuntimeError('CUDA out of memory')
        return device
    results = t._run_model_stage('diarization', {'mic': (Path('mic'), None), 'sys': (Path('sys'), None)}, 'cuda', load, run)
    assert results == {'mic': 'cuda', 'sys': 'cpu'}
    assert calls == [('mic', 'cuda'), ('sys', 'cuda'), ('sys', 'cpu')]
    assert all(ref() is None for ref in references)


def test_load_oom_retries_cpu(runtime, capsys):
    loads = []
    def load(device):
        loads.append(device)
        if device == 'cuda':
            raise RuntimeError('CUDA out of memory loading weights')
        return True
    assert t._run_model_stage('diarization', {'mic': (Path('mic'), None)}, 'cuda', load, lambda *args: 'ok') == {'mic': 'ok'}
    assert loads == ['cuda', 'cpu']
    stderr = capsys.readouterr().err
    assert 'phase=diarization:load:mic' in stderr
    assert 'Traceback' in stderr
    assert 'loading weights' in stderr


def test_alignment_preserves_raw_segments_for_cpu_retry(runtime, monkeypatch):
    original = {'segments': [{'text': 'original'}]}
    wx = sys.modules['whisperx']
    wx.load_align_model = lambda **kwargs: (object(), {})
    seen = []
    def align(segments, model, metadata, audio, device, **kwargs):
        seen.append(segments[0]['text'])
        segments[0]['text'] = 'modified'
        if device == 'cuda':
            raise RuntimeError('CUDA out of memory')
        return {'segments': segments}
    wx.align = align
    result = t._run_model_stage('alignment', {'mic': (Path('mic'), original)}, 'cuda', lambda dev: t.load_alignment_model(dev, 'sv'),
        lambda path, previous, model, device: t.align_transcription(path, previous, device, model))
    assert seen == ['original', 'original']
    assert original['segments'][0]['text'] == 'original'
    assert result['mic']['segments'][0]['text'] == 'modified'


def test_part_releases_whisper_before_alignment_and_keeps_devices_independent(runtime, monkeypatch):
    wx = sys.modules['whisperx']
    references, asr_calls, stages = [], [], []
    class Whisper:
        def transcribe(self, audio, **kwargs):
            asr_calls.append(audio)
            return {'segments': [{'start': 0, 'end': 1, 'text': audio}]}
    def load(*args, **kwargs):
        model = Whisper()
        references.append(weakref.ref(model))
        return model
    wx.load_model = load
    monkeypatch.setattr(t, '_activate_cuda_library_path', lambda: None)
    monkeypatch.setattr(t, 'get_hf_token', lambda: None)
    monkeypatch.setattr(t, 'is_silent', lambda path: False)
    monkeypatch.setattr(t, '_cleanup_pcm_wav', lambda path: None)
    def align(path, result, device, language):
        assert references[0]() is None
        assert asr_calls == ['mic', 'sys']
        stages.append(('align', str(path), device))
        if device == 'cuda':
            raise RuntimeError('CUDA out of memory')
        return result
    monkeypatch.setattr(t, 'load_alignment_model', lambda device, language: True)
    monkeypatch.setattr(t, 'align_transcription', align)
    def diarizer(name, device, token):
        stages.append(('diarizer', device))
        return True
    monkeypatch.setattr(t, 'load_diarizer', diarizer)
    monkeypatch.setattr(t, 'diarize', lambda path, result, *args: result)
    t.process_part(Path('mic'), Path('sys'), part_index=1, use_part_suffix=False, language='sv')
    assert asr_calls == ['mic', 'sys']
    assert stages == [('align', 'mic', 'cuda'), ('align', 'mic', 'cpu'), ('align', 'sys', 'cpu'), ('diarizer', 'cuda')]


def test_diarizer_load_skip_does_not_swallow_cpu_oom(runtime):
    tracks = {'mic': (Path('mic'), {'segments': [{'text': 'kept'}]})}
    def unavailable(device):
        raise RuntimeError('missing token')
    assert t._run_model_stage('diarization', tracks, 'cuda', unavailable, None,
        skip_load_errors=True) == {'mic': tracks['mic'][1]}
    def oom(device):
        raise RuntimeError('out of memory')
    with pytest.raises(RuntimeError, match='out of memory'):
        t._run_model_stage('diarization', tracks, 'cpu', oom, None, skip_load_errors=True)


def test_empty_alignment_does_not_load_model(runtime):
    empty = {'segments': []}
    def load(device):
        raise AssertionError('Empty tracks must not load aligner')
    assert t._run_model_stage('alignment', {'mic': (Path('mic'), empty)}, 'cuda', load, None,
        skip_empty=True) == {'mic': empty}


def test_release_oom_traceback_releases_models_in_exception_chain(runtime):
    import gc

    references = []
    class Model:
        pass
    def underlying_failure():
        model = Model()
        references.append(weakref.ref(model))
        raise RuntimeError('CUDA out of memory in model')
    def wrapped_failure():
        try:
            underlying_failure()
        except RuntimeError as cause:
            raise RuntimeError('CUDA out of memory in wrapper') from cause

    try:
        wrapped_failure()
    except RuntimeError as exc:
        cause = exc.__cause__
        assert cause.__traceback__ is not None
        assert references[0]() is not None
        t._release_oom_traceback(exc)
        gc.collect()
        assert exc.__traceback__ is None
        assert cause.__traceback__ is None
        assert references[0]() is None
