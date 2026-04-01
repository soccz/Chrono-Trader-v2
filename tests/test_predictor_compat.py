import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predictor import (
    _align_sequence_to_model_input,
    _detect_legacy_checkpoint_issues,
    _resolve_model_paths,
)
from utils.config import config


class _DummyModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim


def test_align_sequence_to_model_input_truncates_legacy_width():
    x = torch.randn(1, 4, 32)
    out = _align_sequence_to_model_input(x, _DummyModel(27))
    assert out.shape == (1, 4, 27)
    assert torch.allclose(out, x[..., :27])


def test_align_sequence_to_model_input_pads_when_model_expects_more():
    x = torch.randn(1, 4, 27)
    out = _align_sequence_to_model_input(x, _DummyModel(32))
    assert out.shape == (1, 4, 32)
    assert torch.allclose(out[..., :27], x)
    assert torch.count_nonzero(out[..., 27:]) == 0


def test_align_sequence_to_model_input_warns_once_per_shape_pair():
    x = torch.randn(1, 4, 32)
    warned = set()
    out1 = _align_sequence_to_model_input(x, _DummyModel(27), warn_once_keys=warned)
    out2 = _align_sequence_to_model_input(x, _DummyModel(27), warn_once_keys=warned)
    assert out1.shape == out2.shape == (1, 4, 27)
    assert warned == {(32, 27)}


def test_resolve_model_paths_respects_configured_ensemble_limit(monkeypatch):
    monkeypatch.setattr(
        "inference.predictor.list_numbered_model_paths",
        lambda: [
            "models/model_3.pth",
            "models/model_1.pth",
            "models/model_2.pth",
            "models/model_4.pth",
        ],
    )
    monkeypatch.setattr(config.Gan, "N_ENSEMBLE_MODELS", 3, raising=False)

    paths = _resolve_model_paths()

    assert paths == [
        "models/model_1.pth",
        "models/model_2.pth",
        "models/model_3.pth",
    ]


class _LegacyDecoder:
    pass


class _LegacyCheckpoint:
    input_dim = 27
    decoder = _LegacyDecoder()
    decoder_mode = None
    pe_mode = None


def test_detect_legacy_checkpoint_issues_flags_old_artifact():
    issues = _detect_legacy_checkpoint_issues(_LegacyCheckpoint())

    assert any("input_dim=27" in issue for issue in issues)
    assert any("expected CVAEDecoder" in issue for issue in issues)
    assert any("decoder_mode metadata missing" == issue for issue in issues)
    assert any("pe_mode metadata missing" == issue for issue in issues)


def test_resolve_model_paths_prefers_runtime_compatible_metadata(monkeypatch):
    monkeypatch.setattr(
        "inference.predictor.list_numbered_model_paths",
        lambda: [
            "models/model_1.pth",
            "models/model_2.pth",
            "models/model_3.pth",
        ],
    )
    monkeypatch.setattr(
        "inference.predictor.build_model_artifact_audit",
        lambda: {
            "models": [
                {"path": "models/model_1.pth", "model_key": "model_1", "runtime_compatible": False, "issues": ["input_dim mismatch"]},
                {"path": "models/model_2.pth", "model_key": "model_2", "runtime_compatible": True, "issues": []},
                {"path": "models/model_3.pth", "model_key": "model_3", "runtime_compatible": True, "issues": []},
            ]
        },
    )
    monkeypatch.setattr(config.Gan, "N_ENSEMBLE_MODELS", 3, raising=False)

    paths = _resolve_model_paths()

    assert paths == ["models/model_2.pth", "models/model_3.pth"]
