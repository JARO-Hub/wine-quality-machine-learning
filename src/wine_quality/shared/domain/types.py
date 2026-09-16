import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IndexArray = NDArray[np.int64]
type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type ParameterValue = str | int | float | bool | None
