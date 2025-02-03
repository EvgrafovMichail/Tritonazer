from collections.abc import Iterable, Iterator
from typing import overload

from .tensor_config import TritonTensorConfig

class TritonModelConfig:
    path: str
    name: str
    max_batch_size: int
    platfrom: str
    inputs: Iterator[TritonTensorConfig]
    outputs: Iterator[TritonTensorConfig]

    @overload
    def __init__(
        self,
        path: str,
        name: str,
        inputs: TritonTensorConfig,
        outputs: TritonTensorConfig,
    ) -> None: ...
    @overload
    def __init__(
        self,
        path: str,
        name: str,
        inputs: Iterable[TritonTensorConfig],
        outputs: Iterable[TritonTensorConfig],
    ) -> None: ...
    @overload
    def __init__(
        self,
        path: str,
        name: str,
        inputs: TritonTensorConfig,
        outputs: TritonTensorConfig,
        max_batch_size: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        path: str,
        name: str,
        inputs: Iterable[TritonTensorConfig],
        outputs: Iterable[TritonTensorConfig],
        max_batch_size: int,
    ) -> None: ...
    @overload
    def add_inputs(self, inputs: TritonTensorConfig) -> None: ...
    @overload
    def add_inputs(self, inputs: Iterable[TritonTensorConfig]) -> None: ...
    @overload
    def add_outputs(self, outputs: TritonTensorConfig) -> None: ...
    @overload
    def add_outputs(self, outputs: Iterable[TritonTensorConfig]) -> None: ...
    @overload
    def remove_inputs(self, inputs: str) -> None: ...
    @overload
    def remove_inputs(self, inputs: Iterable[str]) -> None: ...
    @overload
    def remove_outputs(self, outputs: str) -> None: ...
    @overload
    def remove_outputs(self, outputs: Iterable[str]) -> None: ...
    def copy(self) -> TritonModelConfig: ...
