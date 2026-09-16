from typing import cast

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.shared.ports.regressor import Regressor


def _scaled(estimator: object) -> Regressor:
    return cast(Regressor, Pipeline([("scaler", StandardScaler()), ("model", estimator)]))


def _unscaled(estimator: object) -> Regressor:
    return cast(Regressor, Pipeline([("scaler", "passthrough"), ("model", estimator)]))


def model_catalog(seed: int) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            "ols", "Regresión lineal OLS", "regression", lambda: _scaled(LinearRegression()), {}
        ),
        ModelSpec(
            "ridge",
            "Regresión Ridge",
            "regression",
            lambda: _scaled(Ridge(solver="svd")),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        ModelSpec(
            "decision_tree",
            "Árbol CART",
            "trees",
            lambda: _unscaled(DecisionTreeRegressor(random_state=seed, criterion="squared_error")),
            {"model__max_depth": [3, 5, None], "model__min_samples_leaf": [5, 15]},
        ),
        ModelSpec(
            "random_forest",
            "Bosque aleatorio",
            "trees",
            lambda: _unscaled(
                RandomForestRegressor(
                    n_estimators=160,
                    random_state=seed,
                    n_jobs=1,
                    criterion="squared_error",
                )
            ),
            {
                "model__max_depth": [8, None],
                "model__min_samples_leaf": [2, 5],
                "model__max_features": [1.0, "sqrt"],
            },
        ),
        ModelSpec(
            "svr_linear",
            "SVR lineal",
            "svm",
            lambda: _scaled(SVR(kernel="linear", cache_size=64)),
            {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.1, 0.2]},
        ),
        ModelSpec(
            "svr_rbf",
            "SVR RBF",
            "svm",
            lambda: _scaled(SVR(kernel="rbf", cache_size=64)),
            {
                "model__C": [1.0, 10.0, 100.0],
                "model__epsilon": [0.1, 0.2],
                "model__gamma": ["scale", 0.01, 0.1],
            },
        ),
    )
