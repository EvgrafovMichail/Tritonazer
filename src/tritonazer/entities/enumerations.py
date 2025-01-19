from enum import Enum
from typing import Optional


class TritonDataTypes(Enum):
    BOOL = "TYPE_BOOL"
    UINT8 = "TYPE_UINT8"
    UINT16 = "TYPE_UINT16"
    UINT32 = "TYPE_UINT32"
    UINT64 = "TYPE_UINT64"
    INT8 = "TYPE_INT8"
    INT16 = "TYPE_INT16"
    INT32 = "TYPE_INT32"
    INT64 = "TYPE_INT64"
    FP16 = "TYPE_FP16"
    FP32 = "TYPE_FP32"
    FP64 = "TYPE_FP64"
    STRING = "TYPE_STRING"

    @classmethod
    def _missing_(cls, dtype_name: object) -> Optional["TritonDataTypes"]:
        if not isinstance(dtype_name, str):
            return None

        dtype_name = dtype_name.upper()
        prefix_required = "TYPE_"

        if not dtype_name.startswith(prefix_required):
            dtype_name = f"{prefix_required}{dtype_name}"

        for member in cls.__members__.values():
            if dtype_name == member.value:
                return TritonDataTypes(dtype_name)

        return None
