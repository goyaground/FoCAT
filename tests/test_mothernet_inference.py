from __future__ import annotations

import numpy as np
import pytest
import torch

from ticl.model_builder import get_model
from ticl.model_configs import get_model_default_config


@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    config = get_model_default_config("mothernet")
    config["device"] = "cpu"
    config["orchestration"] = {
        "experiment": "mothernet inference tests",
        "progress_bar": False,
    }
    config["prior"]["num_features"] = 3
    config["transformer"].update(
        {
            "emsize": 16,
            "nlayers": 1,
            "nhid_factor": 2,
            "nhead": 2,
            "y_encoder": "linear",
        }
    )
    config["mothernet"].update(
        {
            "weight_embedding_rank": 2,
            "predicted_hidden_layer_size": 8,
            "decoder_embed_dim": 16,
            "decoder_hidden_size": 32,
            "predicted_hidden_layers": 1,
            "decoder_hidden_layers": 1,
            "bins_output_n": 16,
            "bins_to_sum_n": 2,
        }
    )
    _loss, model, _loader, _epoch = get_model(
        config,
        device="cpu",
        should_train=False,
        verbose=False,
    )
    return model


SUPPORT_X = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.25, 0.5],
        [0.75, 0.25],
    ],
    dtype=np.float32,
)
SUPPORT_Y = np.array([0.0, 1.0, 0.5, 1.5], dtype=np.float32)
TREATMENT = np.array([0, 1, 0, 1], dtype=np.int64)


def test_two_row_support_avoids_zero_quantiles(tiny_model) -> None:
    X = SUPPORT_X[:2]
    tiny_model.fit(X, SUPPORT_Y[:2], TREATMENT[:2])

    prediction = tiny_model.predict(X[:1])

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_three_dimensional_fit_predict_keeps_task_axis(tiny_model) -> None:
    X = np.stack([SUPPORT_X, SUPPORT_X + 0.25], axis=1)
    y = np.stack([SUPPORT_Y, SUPPORT_Y + 1.0], axis=1)
    treatment = np.stack([TREATMENT, 1 - TREATMENT], axis=1)

    tiny_model.fit(X, y, treatment)
    prediction = tiny_model.predict(X[:3])

    assert prediction.shape == (3, 2)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize(
    ("X", "y", "treatment", "message"),
    [
        (SUPPORT_X[:, 0], SUPPORT_Y, TREATMENT, "2-D or 3-D"),
        (SUPPORT_X, SUPPORT_Y[:-1], TREATMENT, "shape"),
        (SUPPORT_X, SUPPORT_Y, TREATMENT[:-1], "shape"),
        (
            np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32),
            SUPPORT_Y[:2],
            TREATMENT[:2],
            "finite",
        ),
        (SUPPORT_X, np.array([0.0, 1.0, np.inf, 2.0]), TREATMENT, "finite"),
        (SUPPORT_X, SUPPORT_Y, np.array([0, 1, 2, 0]), "binary"),
        (SUPPORT_X, SUPPORT_Y, np.zeros(4, dtype=np.int64), "both treatment arms"),
        (np.zeros((4, 4)), SUPPORT_Y, TREATMENT, "at most 3 features"),
    ],
)
def test_invalid_support_is_rejected(
    tiny_model,
    X: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tiny_model.fit(X, y, treatment)


def test_three_dimensional_support_requires_both_arms_in_every_task(tiny_model) -> None:
    X = np.stack([SUPPORT_X, SUPPORT_X], axis=1)
    y = np.stack([SUPPORT_Y, SUPPORT_Y], axis=1)
    treatment = np.stack([TREATMENT, np.ones(4, dtype=np.int64)], axis=1)

    with pytest.raises(ValueError, match="both treatment arms"):
        tiny_model.fit(X, y, treatment)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (np.zeros((2, 1), dtype=np.float32), "same 2 features"),
        (np.array([[0.0, np.inf]], dtype=np.float32), "finite"),
        (np.zeros((2, 1, 2), dtype=np.float32), "same dimensionality"),
    ],
)
def test_invalid_query_is_rejected(tiny_model, query: np.ndarray, message: str) -> None:
    tiny_model.fit(SUPPORT_X, SUPPORT_Y, TREATMENT)

    with pytest.raises(ValueError, match=message):
        tiny_model.predict(query)
