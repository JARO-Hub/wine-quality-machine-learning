from typing import Final

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
)
TARGET_NAME: Final[str] = "quality"
