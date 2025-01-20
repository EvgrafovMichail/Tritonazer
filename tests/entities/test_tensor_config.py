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
                data_type=TritonDataTypes.UINT8,
                dims=[-1],
            )

    def test_empty_name_raise_valueerror(self) -> None:
        with pytest.raises(ValueError):
            TritonTensorConfig(
                name="",
                data_type=TritonDataTypes.UINT8,
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
                data_type=TritonDataTypes.UINT8,
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
                [2, 1, -2],
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
                data_type=TritonDataTypes.UINT8,
                dims=dims,
            )

    def test_success_name_init(self) -> None:
        name_expected = "name"

        tensor_config = TritonTensorConfig(
            name="name",
            data_type=TritonDataTypes.UINT8,
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
            pytest.param(
                [1, -1, 5, -1],
                [1, -1, 5, -1],
                id="list-of-mixed-value",
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
            data_type=TritonDataTypes.UINT8,
            dims=dims,
        )

        assert isinstance(tensor_config.dims, list)
        assert tensor_config.dims == dims_expected


class TestNameAttribute:
    TENSOR_CONFIG: TritonTensorConfig = TritonTensorConfig(
        name="name",
        data_type=TritonDataTypes.UINT8,
        dims=[-1],
    )

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
    def test_name_reset_raise_typeerror(
        self,
        name: str,
    ) -> None:
        with pytest.raises(TypeError):
            self.TENSOR_CONFIG.name = name

    def test_name_reset_raise_valueerror(self) -> None:
        with pytest.raises(ValueError):
            self.TENSOR_CONFIG.name = ""

    def test_name_reset_success(self) -> None:
        tensor_config = TritonTensorConfig(
            name="name1",
            data_type=TritonDataTypes.UINT8,
            dims=[-1],
        )
        name_expected = "name2"

        tensor_config.name = "name2"

        assert tensor_config.name == name_expected


class TestDataTypeAttribute:
    @pytest.mark.parametrize(
        "data_type",
        (
            pytest.param(
                float,
                id="invalid-type",
            ),
            pytest.param(
                "invalid",
                id="invalid-value",
            ),
        ),
    )
    def test_data_type_reset_raise_valueerror(
        self,
        data_type: type | str,
    ) -> None:
        tensor_config = TritonTensorConfig(
            name="name",
            data_type=TritonDataTypes.UINT8,
            dims=[-1],
        )

        with pytest.raises(ValueError):
            tensor_config.data_type = data_type

    @pytest.mark.parametrize(
        "data_type,data_type_expected",
        (
            pytest.param(
                "TYPE_INT32",
                TritonDataTypes.INT32,
                id="enum-value",
            ),
            pytest.param(
                "fp64",
                TritonDataTypes.FP64,
                id="user-friendly-value",
            ),
            pytest.param(
                TritonDataTypes.BOOL,
                TritonDataTypes.BOOL,
                id="enum-member",
            ),
        ),
    )
    def test_data_type_reset_success(
        self,
        data_type: str | TritonDataTypes,
        data_type_expected: TritonDataTypes,
    ) -> None:
        tensor_config = TritonTensorConfig(
            name="name",
            data_type=TritonDataTypes.UINT8,
            dims=[-1],
        )

        tensor_config.data_type = data_type

        assert isinstance(tensor_config.data_type, TritonDataTypes)
        assert tensor_config.data_type is data_type_expected


class TestDimsAttribute:
    TENSOR_CONFIG: TritonTensorConfig = TritonTensorConfig(
        name="name",
        data_type=TritonDataTypes.UINT8,
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
    def test_dims_reset_raise_typeerror(
        self,
        dims: Any,
    ) -> None:
        with pytest.raises(TypeError):
            self.TENSOR_CONFIG.dims = dims

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
                [2, 1, -2],
                id="invalid-negative-elem-value",
            ),
        ),
    )
    def test_dims_reset_raise_valueerror(
        self,
        dims: Any,
    ) -> None:
        with pytest.raises(ValueError):
            self.TENSOR_CONFIG.dims = dims

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
    def test_dims_reset_success(
        self,
        dims: Sequence[int],
        dims_expected: list[int],
    ) -> None:
        tensor_config = TritonTensorConfig(
            name="name",
            data_type=TritonDataTypes.STRING,
            dims=[-1],
        )

        tensor_config.dims = dims

        assert isinstance(tensor_config.dims, list)
        assert tensor_config.dims == dims_expected


class TestCopy:
    def test_success(self) -> None:
        tensor_config = TritonTensorConfig(
            name="name",
            data_type=TritonDataTypes.BOOL,
            dims=[-1],
        )
        config_copy = tensor_config.copy()

        assert isinstance(config_copy, TritonTensorConfig)
        assert config_copy is not tensor_config
        assert config_copy.name == tensor_config.name
        assert config_copy.data_type == tensor_config.data_type
        assert config_copy.dims == tensor_config.dims
        assert config_copy.dims is not tensor_config.dims
