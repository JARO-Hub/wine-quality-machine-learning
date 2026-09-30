from typing import cast

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.shared.domain.neural_network import NeuralNetwork
from wine_quality.shared.domain.types import FloatArray, JsonObject
from wine_quality.shared.ports.regressor import Regressor


def neural_specification(seed: int) -> ModelSpec:
    return ModelSpec(
        "mlp_tanh_8",
        "RNA 11-8-1 tanh",
        "neural",
        lambda: cast(
            Regressor,
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(8,),
                            activation="tanh",
                            solver="sgd",
                            learning_rate="constant",
                            learning_rate_init=0.01,
                            momentum=0.0,
                            nesterovs_momentum=False,
                            batch_size=64,
                            max_iter=1000,
                            alpha=0.001,
                            tol=1e-5,
                            n_iter_no_change=50,
                            early_stopping=False,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        {},
    )


def extract_network(estimator: Regressor, training_features: FloatArray) -> NeuralNetwork:
    pipeline = cast(Pipeline, estimator)
    scaler, model = pipeline.named_steps["scaler"], pipeline.named_steps["model"]
    return NeuralNetwork(
        np.array(scaler.mean_, dtype=np.float64),
        np.array(scaler.scale_, dtype=np.float64),
        np.array(model.coefs_[0], dtype=np.float64),
        np.array(model.intercepts_[0], dtype=np.float64),
        np.array(model.coefs_[1], dtype=np.float64),
        np.array(model.intercepts_[1], dtype=np.float64),
        training_features.min(axis=0),
        training_features.max(axis=0),
    )


def learning_summary(estimator: Regressor) -> JsonObject:
    model = cast(Pipeline, estimator).named_steps["model"]
    return {
        "architecture": [11, 8, 1],
        "hidden_activation": "tanh",
        "output_activation": str(model.out_activation_),
        "solver": str(model.solver),
        "learning_rate_init": float(model.learning_rate_init),
        "batch_size": int(model.batch_size),
        "alpha": float(model.alpha),
        "momentum": float(model.momentum),
        "early_stopping": bool(model.early_stopping),
        "max_iter": int(model.max_iter),
        "tol": float(model.tol),
        "n_iter_no_change": int(model.n_iter_no_change),
        "epochs": int(model.n_iter_),
        "hit_epoch_limit": bool(model.n_iter_ >= model.max_iter),
        "loss_curve": [float(loss) for loss in model.loss_curve_],
        "parameter_count": int(sum(a.size for a in (*model.coefs_, *model.intercepts_))),
    }
