"""vlayout：GPU 顶点属性交错布局工具包。

对外暴露 :class:`LayoutError` 与 8 个布局函数，详见
:mod:`vlayout.interleave` 的模块文档与各函数 docstring。
"""

from .interleave import (
    BASE_FORMATS,
    DEFAULT_ALIGN,
    LayoutError,
    attr_size,
    deinterleave,
    interleave,
    layout_batch,
    normalize_type,
    offsets,
    stride,
    validate,
)

__all__ = [
    "BASE_FORMATS",
    "DEFAULT_ALIGN",
    "LayoutError",
    "attr_size",
    "deinterleave",
    "interleave",
    "layout_batch",
    "normalize_type",
    "offsets",
    "stride",
    "validate",
]

__version__ = "0.1.0"
