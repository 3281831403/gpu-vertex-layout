"""gpu-vertex-layout 的验收用例（初始全红）。

``vlayout/interleave.py`` 的 8 个函数目前都是 ``NotImplementedError``，
所以本文件 15 条用例在初始状态下全部失败；修复实现后应全部通过。
本文件是评测标准，不要修改。
"""

import pytest

from vlayout.interleave import (
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


def _flat(vertices, name):
    """把逐顶点字典里的某个属性摊平成 interleave 需要的扁平序列。"""
    values = []
    for vertex in vertices:
        component = vertex[name]
        if isinstance(component, tuple):
            values.extend(component)
        else:
            values.append(component)
    return values


def test_defect_01_attr_size_counts_components():
    """缺陷 1：attr_size 只算单个分量、忽略 count。"""
    assert attr_size("float32", 1) == 4
    assert attr_size("float32", 3) == 12
    assert attr_size("float16", 2) == 4
    assert attr_size("uint8n", 4) == 4


def test_defect_02_stride_rounds_up_to_align():
    """缺陷 2：stride 不按 align 向上取整，末尾属性写到 stride 之外。"""
    schema = [
        {"name": "flag", "format": "uint8", "count": 1},
        {"name": "pos", "format": "float32", "count": 1},
    ]  # 原始字节数 5
    assert stride(schema, 4) == 8
    assert stride(schema, 8) == 8
    assert stride(schema, 16) == 16
    assert stride(schema, 16) % 16 == 0


def test_defect_03_offsets_follow_declaration_order():
    """缺陷 3：offsets 按格式大小排序而不是声明顺序，属性整体错位。"""
    schema = [
        {"name": "flag", "format": "uint8", "count": 1},
        {"name": "pos", "format": "float32", "count": 3},
        {"name": "id", "format": "uint16", "count": 1},
    ]
    assert offsets(schema) == [0, 1, 13]


def test_defect_04_interleave_is_little_endian():
    """缺陷 4：交错写大端，GPU 按小端读到的数值全错。"""
    float_schema = [{"name": "x", "format": "float32", "count": 1}]
    packed = interleave({"x": [1.0]}, float_schema)
    assert packed == b"\x00\x00\x80\x3f"

    short_schema = [{"name": "u", "format": "uint16", "count": 1}]
    packed = interleave({"u": [0x1234]}, short_schema)
    assert packed[:2] == b"\x34\x12"
    assert len(packed) == stride(short_schema, DEFAULT_ALIGN) == 4


def test_defect_05_deinterleave_checks_length():
    """缺陷 5：deinterleave 不校验长度，会读到下一个顶点的字节。"""
    schema = [{"name": "x", "format": "float32", "count": 1}]
    two_vertices = interleave({"x": [1.0, 2.0]}, schema)
    assert len(two_vertices) == 8

    with pytest.raises(LayoutError):
        deinterleave(two_vertices, schema, 1)
    with pytest.raises(LayoutError):
        deinterleave(two_vertices[:-1], schema, 2)


def test_defect_06_normalize_divisor_is_max_integer():
    """缺陷 6：归一化除数用 256，最大值除以它到不了 1。"""
    assert normalize_type("uint8n") == ("uint8", 255)
    assert normalize_type("int8n") == ("int8", 127)
    assert normalize_type("uint16n") == ("uint16", 65535)

    base, divisor = normalize_type("uint8n")
    assert base == "uint8"
    assert 255 / divisor == 1.0
    assert normalize_type("float32") == ("float32", 1)


def test_defect_07_unknown_format_is_rejected():
    """缺陷 7：未知格式被静默当成 float32。"""
    with pytest.raises(LayoutError):
        attr_size("float31", 1)
    with pytest.raises(LayoutError):
        stride([{"name": "x", "format": "vec3", "count": 1}], 4)

    problems = validate([{"name": "x", "format": "float31", "count": 1}], 4)
    assert problems
    assert any("float31" in problem for problem in problems)


def test_defect_08_validate_reports_stride_mismatch():
    """缺陷 8：validate 不查 stride 是否整除长度，二次对齐后长度对不上。"""
    schema = [{"name": "pos", "format": "float32", "count": 2}]  # 8 字节，stride 8
    assert validate(schema, 0) == []
    assert validate(schema, 8) == []
    assert validate(schema, 16) == []

    problems = validate(schema, 12)
    assert problems
    assert any("stride" in problem.lower() for problem in problems)


def test_defect_09_batch_buffers_do_not_mix():
    """缺陷 9：批量共用同一输出缓冲，批次之间的字节互相串。"""
    schema = [{"name": "pos", "format": "float32", "count": 2}]
    first = {"pos": [1.0, 2.0]}
    second = {"pos": [3.0, 4.0, 5.0, 6.0]}

    result = layout_batch([first, second], schema, {"align": 4})
    assert result["stride"] == 8
    assert result["counts"] == [1, 2]
    assert result["offsets"] == [0, 8]
    assert len(result["data"]) == 24

    assert deinterleave(result["data"][:8], schema, 1) == [{"pos": (1.0, 2.0)}]
    assert deinterleave(result["data"][8:], schema, 2) == [
        {"pos": (3.0, 4.0)},
        {"pos": (5.0, 6.0)},
    ]

    with pytest.raises(LayoutError):
        layout_batch([first, {"pos": [7.0, 8.0, 9.0]}], schema, {"align": 4})
    assert first == {"pos": [1.0, 2.0]}


def test_defect_10_batch_rejects_zero_align():
    """缺陷 10：align=0 不校验，stride 计算时直接除零。"""
    schema = [{"name": "pos", "format": "float32", "count": 2}]
    with pytest.raises(LayoutError):
        layout_batch([{"pos": [1.0, 2.0]}], schema, {"align": 0})
    with pytest.raises(LayoutError):
        layout_batch([{"pos": [1.0, 2.0]}], schema, {"align": -4})


def test_attr_size_single_float32_is_four_bytes():
    """附加：单个 float32 属性只占 4 字节。"""
    assert attr_size("float32", 1) == 4


def test_stride_rejects_zero_align():
    """附加：stride 必须拒绝 align=0，不能除零。"""
    schema = [{"name": "x", "format": "float32", "count": 1}]
    with pytest.raises(LayoutError):
        stride(schema, 0)


def test_empty_schema_yields_empty_layout():
    """附加：空 schema 的各个入口都返回空布局。"""
    assert offsets([]) == []
    assert stride([], DEFAULT_ALIGN) == 0
    assert interleave({}, []) == b""
    assert deinterleave(b"", [], 0) == []
    assert validate([], 0) == []


def test_interleave_deinterleave_roundtrip():
    """附加：交错后再解包，逐顶点、逐属性必须完全一致。"""
    schema = [
        {"name": "pos", "format": "float32", "count": 3},
        {"name": "color", "format": "uint8n", "count": 4},
        {"name": "uv", "format": "float16", "count": 2},
    ]
    vertices = [
        {"pos": (0.0, 1.0, 2.0), "color": (0, 128, 255, 255), "uv": (0.5, 1.0)},
        {"pos": (-1.5, 4.25, 9.0), "color": (1, 2, 3, 4), "uv": (-2.0, 0.25)},
    ]
    attrs = {attr["name"]: _flat(vertices, attr["name"]) for attr in schema}

    data = interleave(attrs, schema)
    assert len(data) == stride(schema, DEFAULT_ALIGN) * len(vertices)
    assert deinterleave(data, schema, len(vertices)) == vertices


def test_batch_all_success_preserves_order():
    """附加：批次全部合法时，输出顺序与 buffers 顺序一致。"""
    schema = [{"name": "id", "format": "uint16", "count": 1}]
    buffers = [
        {"id": [1]},
        {"id": [2, 3]},
        {"id": [4, 5, 6]},
    ]
    result = layout_batch(buffers, schema, {"align": 4})
    assert result["stride"] == 4
    assert result["counts"] == [1, 2, 3]
    assert result["offsets"] == [0, 4, 12]
    assert len(result["data"]) == 24

    vertices = deinterleave(result["data"], schema, 6)
    assert [vertex["id"] for vertex in vertices] == [1, 2, 3, 4, 5, 6]
