from collections.abc import Sequence
from typing import Any

import pytest

from tritonazer.entities.enumerations import TritonDataTypes
from tritonazer.entities.tensor_config import TritonTensorConfig


class TestInit:
    @pytest.mark.parametrize(
        "name",
        (
            pytest.param(
                42,
                id="numeric-type",
            ),
            pytest.param(
                ["name"],
                id="sequence-type",
            ),
        ),
    )
    def test_invalid_name_raise_typeerror(self, name: str) -> None:
        with pytest.raises(TypeError):
            TritonTensorConfig(
                name=name,
                data_type="unit8",
                dims=[-1],
            )

    def test_empty_name_raise_valueerror(self) -> None:
        with pytest.raises(ValueError):
            TritonTensorConfig(
                name="",
                data_type="unit8",
                dims=[-1],
            )

    def test_invalid_dtype_raise_valueerror(self) -> None:
        with pytest.raises(ValueError):
            TritonTensorConfig(
                name="name",
                data_type="invalid-type",
                dims=[-1],
            )

    @pytest.mark.parametrize(
        "dims",
        (
            pytest.param(
                42,
                id="numeric-object-type",
            ),
            pytest.param(
                (i**2 for i in range(3)),
                id="iterable-object-type",
            ),
            pytest.param(
                "string",
                id="string-type",
            ),
            pytest.param(
                (3.14, 2.72, 1),
                id="float-elem-type",
            ),
        ),
    )
    def test_invalid_dims_raise_typeerror(
        self,
        dims: Any,
    ) -> None:
        with pytest.raises(TypeError):
            TritonTensorConfig(
                name="name",
                data_type="uint8",
                dims=dims,
            )

    @pytest.mark.parametrize(
        "dims",
        (
            pytest.param(
                [],
                id="empty-object",
            ),
            pytest.param(
                (2, 1, 0),
                id="zero-elem-value",
            ),
            pytest.param(
                [2, 1, -1],
                id="invalid-negative-elem-value",
            ),
        ),
    )
    def test_invalid_dims_raise_valueerror(
        self,
        dims: Any,
    ) -> None:
        with pytest.raises(ValueError):
            TritonTensorConfig(
                name="name",
                data_type="unit8",
                dims=dims,
            )

    def test_success_name_init(self) -> None:
        name_expected = "name"

        tensor_config = TritonTensorConfig(
            name="name",
            data_type="uint8",
            dims=[-1],
        )

        assert tensor_config.name == name_expected

    @pytest.mark.parametrize(
        "data_type,data_type_expected",
        (
            pytest.param(
                "TYPE_FP16",
                TritonDataTypes.FP16,
                id="enum-value",
            ),
            pytest.param(
                "uint8",
                TritonDataTypes.UINT8,
                id="user-friendly-value",
            ),
            pytest.param(
                TritonDataTypes.STRING,
                TritonDataTypes.STRING,
                id="enum-member",
            ),
        ),
    )
    def test_success_data_type_init(
        self,
        data_type: TritonDataTypes | str,
        data_type_expected: TritonDataTypes,
    ) -> None:
        tensor_config = TritonTensorConfig(
            name="name",
            data_type=data_type,
            dims=[-1],
        )

        assert isinstance(tensor_config.data_type, TritonDataTypes)
        assert tensor_config.data_type is data_type_expected

    @pytest.mark.parametrize(
        "dims,dims_expected",
        (
            pytest.param(
                (1, 2, 3),
                [1, 2, 3],
                id="tuple-of-positives",
            ),
            pytest.param(
                [-1, -1],
                [-1, -1],
                id="list-of-negatives",
            ),
        ),
    )
    def test_success_dims_init(
        self,
        dims: Sequence[int],
        dims_expected: list[int],
    ) -> None:
        tensor_config = TritonTensorConfig(
            name="name",
            data_type="uint8",
            dims=dims,
        )

        assert isinstance(tensor_config.dims, list)
        assert tensor_config.dims == dims_expected
