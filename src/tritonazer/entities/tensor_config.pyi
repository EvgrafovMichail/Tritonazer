from collections.abc import Sequence
from typing import overload

from .enumerations import TritonDataTypes

class TritonTensorConfig:
    name: str
    data_type: TritonDataTypes
    dims: list[int]

    @overload
    def __init__(
        self,
        name: str,
        data_type: TritonDataTypes,
        dims: Sequence[int],
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: str,
        data_type: str,
        dims: Sequence[int],
    ) -> None: ...
    def copy(self, deep: bool = True) -> TritonTensorConfig: ...
