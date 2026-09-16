from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int = 42
    test_fraction: float = 0.2
    cv_folds: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction debe estar entre 0 y 1.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds debe ser al menos 2.")
