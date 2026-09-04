from argparse import Namespace
from pathlib import Path

import pytest

from mlops.rnnoise_pipeline import make_manifest, recommended_epochs, validate_pcm


def test_recommended_epochs_targets_75000_updates() -> None:
    assert recommended_epochs(200_000, 128) == 49
    with pytest.raises(ValueError, match="full batch"):
        recommended_epochs(10, 128)


def test_pcm_validation_hashes_content_and_rejects_odd_bytes(tmp_path: Path) -> None:
    pcm = tmp_path / "audio.pcm"
    pcm.write_bytes(b"\x00\x01" * 48_000)
    report = validate_pcm(pcm, "audio")
    assert report["duration_seconds"] == 1.0
    assert len(report["sha256"]) == 64

    pcm.write_bytes(b"x")
    with pytest.raises(ValueError, match="headerless int16"):
        validate_pcm(pcm, "audio")


def test_manifest_keeps_train_and_eval_lineage_separate(tmp_path: Path) -> None:
    paths = {}
    for name, content in (("train_speech", b"aa"), ("train_noise", b"bb"),
                          ("eval_speech", b"cc"), ("eval_noise", b"dd")):
        paths[name] = tmp_path / f"{name}.pcm"
        paths[name].write_bytes(content)
    card = tmp_path / "DATA_CARD.md"
    card.write_text("licensed test data")
    args = Namespace(
        **paths,
        data_card=card,
        rir_list=None,
        train_sequences=10_000,
        eval_sequences=100,
        seed=7,
    )
    manifest = make_manifest(args)
    assert manifest["inputs"]["train_speech"]["sha256"] != manifest["inputs"]["eval_speech"]["sha256"]
    assert manifest["feature_generation"]["seed"] == 7
