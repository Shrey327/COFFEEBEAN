from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from coffeebean.cli import (
    RNNOISE_MODEL,
    SPEECH_SAMPLE_RATE,
    enhance_audio,
    evaluate_audio,
    fxlmm_score,
    generate_paper_signal,
    live_demo,
    mix_audio,
    modified_fxlmm_score,
    run_anc,
    stream_demo,
)
from coffeebean.demo import _summary_text

THRESHOLDS = (1.0, 2.0, 4.0)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_hampel_score_boundaries(sign: float) -> None:
    assert fxlmm_score(sign * 0.0, THRESHOLDS) == 0.0
    assert fxlmm_score(sign * 1.0, THRESHOLDS) == sign
    assert fxlmm_score(sign * 2.0, THRESHOLDS) == sign
    assert fxlmm_score(sign * 4.0, THRESHOLDS) == 0.0


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_modified_score_is_continuous_and_redescends(sign: float) -> None:
    assert modified_fxlmm_score(sign * 0.0, THRESHOLDS) == 0.0
    assert modified_fxlmm_score(sign * 1.0, THRESHOLDS) == sign
    assert modified_fxlmm_score(sign * 2.5, THRESHOLDS) == pytest.approx(sign * 0.5)
    assert modified_fxlmm_score(sign * 4.0, THRESHOLDS) == 0.0


def test_fixed_signal_and_simulation_are_reproducible() -> None:
    _, first = generate_paper_signal(duration=1.0, impulse_times=(0.5,))
    _, second = generate_paper_signal(duration=1.0, impulse_times=(0.5,))
    np.testing.assert_array_equal(first, second)
    first_run = run_anc(first, algorithm="modified-fxlmm")
    second_run = run_anc(second, algorithm="modified-fxlmm")
    np.testing.assert_array_equal(first_run["residual"], second_run["residual"])
    assert np.all(np.isfinite(first_run["coefficient_norm"]))


def test_robust_updates_limit_impulse_coefficient_growth() -> None:
    _, reference = generate_paper_signal(
        duration=2.0, impulse_times=(1.0,), impulse_width=0.2
    )
    conventional = run_anc(reference, algorithm="fxlms")
    robust = run_anc(reference, algorithm="modified-fxlmm")
    assert np.max(robust["coefficient_norm"]) <= np.max(
        conventional["coefficient_norm"]
    )


@pytest.mark.parametrize("profile", ("continuous", "impulsive"))
def test_mix_is_48khz_mono_and_hits_requested_snr(tmp_path, profile: str) -> None:
    t = np.arange(SPEECH_SAMPLE_RATE, dtype=np.float64) / SPEECH_SAMPLE_RATE
    clean = 0.1 * np.sin(2 * np.pi * 220 * t)
    clean_path = tmp_path / "clean.wav"
    output_path = tmp_path / "noisy.wav"
    sf.write(clean_path, clean, SPEECH_SAMPLE_RATE, subtype="FLOAT")

    metadata = mix_audio(clean_path, output_path, profile=profile, snr_db=0.0)
    mixed, sample_rate = sf.read(output_path, always_2d=True)
    assert sample_rate == SPEECH_SAMPLE_RATE
    assert mixed.shape == (SPEECH_SAMPLE_RATE, 1)
    assert metadata["actual_snr_db"] == pytest.approx(0.0, abs=1e-9)
    assert np.max(np.abs(mixed)) <= 0.981


def test_mix_accepts_and_tiles_real_noise_recording(tmp_path) -> None:
    clean_rate = SPEECH_SAMPLE_RATE
    noise_rate = 44_100
    clean_t = np.arange(clean_rate, dtype=np.float64) / clean_rate
    noise_t = np.arange(noise_rate // 4, dtype=np.float64) / noise_rate
    clean_path = tmp_path / "clean.wav"
    noise_path = tmp_path / "field-recording.wav"
    output_path = tmp_path / "noisy.wav"
    sf.write(clean_path, 0.1 * np.sin(2 * np.pi * 220 * clean_t), clean_rate)
    sf.write(noise_path, 0.1 * np.sin(2 * np.pi * 900 * noise_t), noise_rate)

    metadata = mix_audio(
        clean_path, output_path, noise_path=noise_path, snr_db=5.0
    )
    mixed, sample_rate = sf.read(output_path, always_2d=True)
    assert sample_rate == SPEECH_SAMPLE_RATE
    assert mixed.shape == (SPEECH_SAMPLE_RATE, 1)
    assert metadata["profile"] == "recording"
    assert metadata["noise"] == str(noise_path)
    assert metadata["actual_snr_db"] == pytest.approx(5.0, abs=1e-9)


def test_mix_requires_exactly_one_noise_source(tmp_path) -> None:
    clean_path = tmp_path / "clean.wav"
    sf.write(clean_path, np.ones(100), SPEECH_SAMPLE_RATE)
    with pytest.raises(ValueError, match="exactly one"):
        mix_audio(clean_path, tmp_path / "out.wav", snr_db=0.0)


def test_evaluate_reports_improvement(tmp_path) -> None:
    t = np.arange(SPEECH_SAMPLE_RATE, dtype=np.float64) / SPEECH_SAMPLE_RATE
    clean = 0.1 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 440 * t))
    generator = np.random.default_rng(7)
    noisy = clean + 0.03 * generator.standard_normal(len(clean))
    enhanced = clean + 0.005 * generator.standard_normal(len(clean))
    paths = {
        name: tmp_path / f"{name}.wav" for name in ("clean", "noisy", "enhanced")
    }
    for name, samples in (
        ("clean", clean),
        ("noisy", noisy),
        ("enhanced", enhanced),
    ):
        sf.write(paths[name], samples, SPEECH_SAMPLE_RATE, subtype="FLOAT")

    output = tmp_path / "metrics.json"
    report = evaluate_audio(
        paths["clean"], paths["noisy"], paths["enhanced"], output
    )
    assert report["improvement"]["si_sdr_db"] > 0
    assert report["improvement"]["stoi"] >= 0
    assert json.loads(output.read_text()) == report


def test_enhance_rejects_non_48khz_input_before_model_load(tmp_path) -> None:
    input_path = tmp_path / "input.wav"
    sf.write(input_path, np.zeros(16_000), 16_000, subtype="FLOAT")
    with pytest.raises(ValueError, match="48 kHz"):
        enhance_audio(input_path, tmp_path / "output.wav")


def test_enhance_routes_rnnoise_v02_and_reports_latency(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    samples = np.linspace(-0.1, 0.1, 960, dtype=np.float32)
    sf.write(input_path, samples, SPEECH_SAMPLE_RATE, subtype="FLOAT")
    monkeypatch.setattr(
        "coffeebean.cli._enhance_rnnoise", lambda value: (value.copy(), 960)
    )

    report = enhance_audio(input_path, output_path, model="RNNoise-v0.2")

    assert report["model"] == RNNOISE_MODEL
    assert report["algorithmic_latency_samples"] == 960
    assert report["algorithmic_latency_ms"] == pytest.approx(20.0)
    output, rate = sf.read(output_path)
    assert rate == SPEECH_SAMPLE_RATE
    np.testing.assert_allclose(output, samples, atol=1e-7)


def test_live_demo_records_then_enhances(tmp_path, monkeypatch) -> None:
    class FakeInputStream:
        latency = 0.01

        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def read(self, frames: int):
            return np.full((frames, 1), 0.1, dtype=np.float32), False

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeInputStream)
    )

    def fake_enhance(input_path, output_path, *, model):
        samples, sample_rate = sf.read(input_path)
        sf.write(output_path, samples * 0.5, sample_rate, subtype="FLOAT")
        return {"model": model, "real_time_factor": 0.1}

    monkeypatch.setattr("coffeebean.cli.enhance_audio", fake_enhance)
    report = live_demo(tmp_path / "live", duration=0.02)
    assert report["streaming"] is False
    assert report["input_overflow_blocks"] == 0
    assert report["duration_seconds"] == pytest.approx(0.02)
    assert report["enhanced"]["peak"] < report["noisy"]["peak"]
    assert (tmp_path / "live" / "report.json").exists()


def test_stream_demo_processes_during_capture(tmp_path, monkeypatch) -> None:
    class FakeInputStream:
        latency = 0.005

        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def read(self, frames: int):
            return np.full((frames, 1), 0.1, dtype=np.float32), False

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeInputStream)
    )
    monkeypatch.setattr(
        "coffeebean.cli._load_stream_enhancer", lambda _model: lambda x: x * 0.5
    )
    report = stream_demo(
        tmp_path / "stream", duration=0.025, chunk_ms=10, context_ms=20
    )
    assert report["streaming"] is True
    assert report["input_overflow_blocks"] == 0
    assert report["processing_deadline_misses"] == 0
    noisy, rate = sf.read(tmp_path / "stream" / "noisy.wav")
    enhanced, _ = sf.read(tmp_path / "stream" / "enhanced.wav")
    assert rate == SPEECH_SAMPLE_RATE
    assert len(noisy) == len(enhanced) == 1_200
    np.testing.assert_allclose(enhanced, noisy * 0.5, atol=1e-6)


def test_demo_summary_surfaces_deadlines_and_latency() -> None:
    summary = _summary_text(
        {
            "duration_seconds": 10.0,
            "sample_rate": 48_000,
            "processing_seconds_median": 0.011,
            "estimated_output_latency_seconds": 0.31,
            "processing_deadline_misses": 0,
            "input_overflow_blocks": 0,
            "noisy": {"peak": 0.9},
            "enhanced": {"peak": 0.7, "clipping_fraction": 0.0},
        }
    )
    assert "Median inference: 11.0 ms" in summary
    assert "Estimated latency: 310 ms" in summary
    assert "Deadline misses: 0" in summary
