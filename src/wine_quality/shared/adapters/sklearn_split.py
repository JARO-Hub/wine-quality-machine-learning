import numpy as np
from sklearn.model_selection import KFold, train_test_split

from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.split_plan import SplitPlan


def build_split(dataset: Dataset, config: ExperimentConfig) -> SplitPlan:
    _, counts = np.unique(dataset.target, return_counts=True)
    if counts.min() < 2:
        raise ValueError("La partición estratificada requiere al menos dos filas de cada quality.")
    train, test = train_test_split(
        np.arange(dataset.rows_clean, dtype=np.int64),
        test_size=config.test_fraction,
        random_state=config.seed,
        stratify=dataset.target,
    )
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if train.size < 2 * config.cv_folds:
        raise ValueError("Cada fold necesita al menos dos filas de validación para calcular R².")
    splitter = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    folds = tuple(
        (np.asarray(fit, dtype=np.int64), np.asarray(validation, dtype=np.int64))
        for fit, validation in splitter.split(train)
    )
    return SplitPlan(train, test, folds)
