from collections.abc import Sequence
from typing import Any

from .enumerations import TritonDataTypes


class TritonTensorConfig:
    _name: str
    _data_type: TritonDataTypes
    _dims: list[int]

    def __init__(
        self,
        name: str,
        data_type: TritonDataTypes | str,
        dims: Sequence[int],
    ) -> None:
        self._name = self._get_valid_name(name)
        self._data_type = TritonDataTypes(data_type)
        self._dims = self._get_valid_dims(dims)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = self._get_valid_name(value)

    @property
    def data_type(self) -> TritonDataTypes:
        return self._data_type

    @data_type.setter
    def data_type(self, value: TritonDataTypes | str) -> None:
        self._data_type = TritonDataTypes(value)

    @property
    def dims(self) -> list[int]:
        return self._dims

    @dims.setter
    def dims(self, value: Sequence[int]) -> None:
        self._dims = self._get_valid_dims(value)

    def copy(self, deep: bool = True) -> "TritonTensorConfig":
        return TritonTensorConfig(
            name=self._name,
            data_type=self._data_type,
            dims=self.dims if deep else self._dims,
        )

    @staticmethod
    def _get_valid_name(name: Any) -> str:
        if not isinstance(name, str):
            raise TypeError(
                "tensor name must be string but value of type "
                f"{type(name).__name__} was given: {name}"
            )

        if not name:
            raise ValueError("tensor name must not be empty")

        return name

    @staticmethod
    def _get_valid_dims(dims: Sequence[int]) -> list[int]:
        if not isinstance(dims, Sequence):
            raise TypeError(
                "tensor dims must be sequence but object of "
                f"type {type(dims).__name__} was given: {dims}"
            )

        if not dims:
            raise ValueError("tensor dims must not be empty")

        dims_valid: list[int] = []

        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError(
                    "tensor dims must contain only integer numbers "
                    "but given dims contains value of type "
                    f"{type(dim).__name__}: {dim}"
                )

            if dim <= 0 and dim != -1:
                raise ValueError(
                    "tensor dims must contain only integer numbers "
                    "greater than 0 or equal to -1 but given dims "
                    f"contains invalid value: {dim}"
                )

            dims_valid.append(dim)

        return dims_valid
