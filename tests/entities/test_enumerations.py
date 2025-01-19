import pytest

from tritonazer.entities.enumerations import TritonDataTypes


class TestTritonDataTypes:
    @pytest.mark.parametrize(
        "value",
        (
            pytest.param(
                42,
                id="invalid-type",
            ),
            pytest.param(
                "invalid-value",
                id="invalid-value",
            ),
            pytest.param(
                "TYPE_",
                id="prefix-only",
            ),
            pytest.param(
                "_uint8",
                id="postfix-only",
            ),
        ),
    )
    def test_enum_conversion_raise_valueerror(
        self,
        value: int | str,
    ) -> None:
        with pytest.raises(ValueError):
            TritonDataTypes(value)

    @pytest.mark.parametrize(
        "value,member_expected",
        (
            pytest.param(
                TritonDataTypes.BOOL,
                TritonDataTypes.BOOL,
                id="enum-value",
            ),
            pytest.param(
                TritonDataTypes.FP16.value,
                TritonDataTypes.FP16,
                id="uppercase",
            ),
            pytest.param(
                "type_uint8",
                TritonDataTypes.UINT8,
                id="lowercase",
            ),
            pytest.param(
                "tYpE_InT16",
                TritonDataTypes.INT16,
                id="mixed_case",
            ),
            pytest.param(
                "string",
                TritonDataTypes.STRING,
                id="ommited-prefix",
            ),
        ),
    )
    def test_enum_conversion_success(
        self,
        value: str,
        member_expected: TritonDataTypes,
    ) -> None:
        member = TritonDataTypes(value)

        assert member is member_expected
