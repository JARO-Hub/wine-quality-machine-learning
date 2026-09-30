import math

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from wine_quality.cases.case_04_covid.config import CovidConfig
from wine_quality.shared.domain.forecast_dataset import ForecastDataset
from wine_quality.shared.domain.split_plan import SplitPlan


def build_temporal_split(dataset: ForecastDataset, config: CovidConfig) -> SplitPlan:
    test_count = math.ceil(dataset.target.size * config.test_fraction)
    train_count = dataset.target.size - test_count
    if test_count < 2 or train_count // (config.cv_folds + 1) < 2:
        raise ValueError("Datos insuficientes para prueba y folds con al menos dos respuestas.")
    train = np.arange(train_count, dtype=np.int64)
    test = np.arange(train_count, dataset.target.size, dtype=np.int64)
    folds = tuple(
        (fit.astype(np.int64), validation.astype(np.int64))
        for fit, validation in TimeSeriesSplit(n_splits=config.cv_folds).split(train)
    )
    return SplitPlan(train, test, folds)
