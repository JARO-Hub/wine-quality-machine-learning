from typing import cast

from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.shared.adapters.sklearn_model_catalog import model_catalog
from wine_quality.shared.ports.regressor import Regressor


def _target_scaled(specification: ModelSpec) -> ModelSpec:
    return ModelSpec(
        specification.identifier,
        specification.label,
        specification.family,
        lambda: cast(
            Regressor,
            TransformedTargetRegressor(
                regressor=specification.build(), transformer=StandardScaler()
            ),
        ),
        {f"regressor__{key}": values for key, values in specification.parameter_grid.items()},
    )


def covid_model_catalog(seed: int) -> tuple[ModelSpec, ...]:
    neural = ModelSpec(
        "mlp_tanh_8",
        "RNA 18-8-1 tanh",
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
                            solver="lbfgs",
                            alpha=0.001,
                            max_iter=2000,
                            max_fun=50000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        {},
    )
    return tuple(_target_scaled(spec) for spec in (*model_catalog(seed), neural))
