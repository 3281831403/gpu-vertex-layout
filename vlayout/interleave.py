"""GPU 顶点属性交错布局工具。

本模块把若干顶点属性（attribute）按声明顺序交错（interleave）进一条紧凑的
顶点缓冲，并支持反向解包。所有读写一律使用小端字节序（little-endian），
不依赖宿主机字节序。

属性统一描述为字典::

    {"name": "position", "format": "float32", "count": 3}

``format`` 支持的基础类型：

==========  ================================
字节数       格式名
==========  ================================
1           int8, uint8
2           int16, uint16, float16
4           int32, uint32, float32
8           int64, uint64, float64
==========  ================================

格式名末尾追加 ``n`` 表示归一化整型（normalized integer），例如
``uint8n``、``int16n``。归一化只允许用于整型，浮点类型加 ``n``
视为未知格式。缓冲区里仍然存放原始整数，``normalize_type`` 给出换算到
[0, 1]（无符号）或 [-1, 1]（有符号）所需的分母。

布局规则：

* 属性内部各分量紧凑排列，不插入 padding；
* 属性偏移按 ``schema`` 的声明顺序依次累加，不按格式大小重排；
* 一个顶点的步长 ``stride`` 等于全部属性字节数之和向上对齐到 ``align``
  的整数倍（``interleave``/``deinterleave``/``validate`` 使用模块默认
  ``DEFAULT_ALIGN = 4``，``layout_batch`` 可通过 config 覆盖）；
* 对齐补零字节只出现在顶点末尾，顶点之间没有空隙。

容差为 0 字节：任何 ``data`` 的长度必须严格等于 ``stride * count``。
"""

import struct

DEFAULT_ALIGN = 4

#: 所有受支持的格式名（不含归一化后缀 ``n``）。
BASE_FORMATS = (
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
)

#: 基础格式 -> (单分量字节数, struct 代码)
_FORMAT_INFO = {
    "int8": (1, "b"),
    "uint8": (1, "B"),
    "int16": (2, "h"),
    "uint16": (2, "H"),
    "int32": (4, "i"),
    "uint32": (4, "I"),
    "int64": (8, "q"),
    "uint64": (8, "Q"),
    "float16": (2, "e"),
    "float32": (4, "f"),
    "float64": (8, "d"),
}

_INT_FORMATS = frozenset(
    fmt for fmt in _FORMAT_INFO if fmt.startswith(("int", "uint"))
)
_UNSIGNED_FORMATS = frozenset(fmt for fmt in _FORMAT_INFO if fmt.startswith("uint"))


class LayoutError(ValueError):
    """顶点布局或顶点缓冲数据不合法。

    ``attr_size``/``stride``/``interleave``/``deinterleave``/
    ``normalize_type``/``layout_batch`` 在输入违反契约时抛出本异常；
    ``validate`` 则不抛异常，而是把问题字符串收集到返回值里。
    """


def _parse_format(fmt):
    """把格式名拆成 ``(基础名, struct 代码, 单分量字节数, 是否归一化)``。

    未知格式、浮点类型加 ``n`` 后缀时抛 :class:`LayoutError`。
    """
    if not isinstance(fmt, str):
        raise LayoutError(f"格式名必须是字符串: {fmt!r}")

    info = _FORMAT_INFO.get(fmt)
    if info is not None:
        size, code = info
        return fmt, code, size, False

    if fmt.endswith("n"):
        base = fmt[:-1]
        if base in _FORMAT_INFO:
            if base not in _INT_FORMATS:
                raise LayoutError(f"归一化后缀 n 只允许整型格式: {fmt!r}")
            size, code = _FORMAT_INFO[base]
            return base, code, size, True

    raise LayoutError(f"未知格式: {fmt!r}")


def _attr_size(fmt, count):
    """``attr_size`` 的内部实现，附带 count 校验。"""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise LayoutError(f"count 必须是 >= 1 的整数: {count!r}")
    _base, _code, size, _normalized = _parse_format(fmt)
    return size * count


def _schema_problems(schema):
    """收集 schema 的结构性问题，不抛异常（类型错误除外）。"""
    if not isinstance(schema, list):
        raise LayoutError(f"schema 必须是属性列表: {schema!r}")

    problems = []
    seen = set()
    for index, attr in enumerate(schema):
        if not isinstance(attr, dict):
            problems.append(f"属性 {index} 必须是字典")
            continue

        missing = [key for key in ("name", "format", "count") if key not in attr]
        if missing:
            problems.append(f"属性 {index} 缺少字段: {', '.join(missing)}")
            continue

        name = attr["name"]
        if not isinstance(name, str) or not name:
            problems.append(f"属性 {index} 的 name 必须是非空字符串: {name!r}")
        elif name in seen:
            problems.append(f"属性名重复: {name!r}")
        else:
            seen.add(name)

        try:
            _parse_format(attr["format"])
        except LayoutError as exc:
            problems.append(f"属性 {name!r}: {exc}")

        count = attr["count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            problems.append(f"属性 {name!r} 的 count 必须是正整数: {count!r}")

    return problems


def _require_schema(schema):
    """严格校验 schema，有问题直接抛 :class:`LayoutError`。"""
    problems = _schema_problems(schema)
    if problems:
        raise LayoutError("; ".join(problems))


def _require_align(align):
    if not isinstance(align, int) or isinstance(align, bool) or align <= 0:
        raise LayoutError(f"align 必须是正整数: {align!r}")
    return align


def _raw_size(schema):
    total = 0
    for attr in schema:
        total += _attr_size(attr["format"], attr["count"])
    return total


def _stride(schema, align):
    _require_align(align)
    _require_schema(schema)
    raw = _raw_size(schema)
    if raw == 0:
        return 0
    return ((raw + align - 1) // align) * align


def _offsets(schema):
    _require_schema(schema)
    result = []
    cursor = 0
    for attr in schema:
        result.append(cursor)
        cursor += _attr_size(attr["format"], attr["count"])
    return result


def _pack(attrs, schema, align):
    """按 ``schema`` 与 ``align`` 打包，返回 ``(bytes, 顶点数)``。"""
    _require_align(align)
    _require_schema(schema)

    if not isinstance(attrs, dict):
        raise LayoutError(f"attrs 必须是字典: {attrs!r}")

    names = [attr["name"] for attr in schema]
    if set(attrs) != set(names):
        missing = sorted(set(names) - set(attrs))
        extra = sorted(set(attrs) - set(names))
        parts = []
        if missing:
            parts.append(f"缺少属性: {missing}")
        if extra:
            parts.append(f"多余属性: {extra}")
        raise LayoutError("attrs 与 schema 的键集合不符（" + "; ".join(parts) + "）")

    if not schema:
        return b"", 0

    columns = []
    vertex_count = None
    for attr in schema:
        base, code, size, _normalized = _parse_format(attr["format"])
        component_count = attr["count"]
        try:
            values = list(attrs[attr["name"]])
        except TypeError as exc:
            raise LayoutError(f"属性 {attr['name']!r} 的值必须是可迭代序列") from exc
        if len(values) % component_count != 0:
            raise LayoutError(
                f"属性 {attr['name']!r} 的长度 {len(values)} "
                f"不是 count {component_count} 的整数倍"
            )
        count = len(values) // component_count
        if vertex_count is None:
            vertex_count = count
        elif count != vertex_count:
            raise LayoutError(
                f"属性 {attr['name']!r} 推导出 {count} 个顶点，"
                f"与其它属性的 {vertex_count} 个不一致"
            )
        columns.append((attr["name"], code, size, component_count, values))

    step = _stride(schema, align)
    starts = _offsets(schema)
    out = bytearray(step * vertex_count)

    for vertex in range(vertex_count):
        base_offset = vertex * step
        for (name, code, size, component_count, values), start in zip(columns, starts):
            for component in range(component_count):
                value = values[vertex * component_count + component]
                try:
                    packed = struct.pack("<" + code, value)
                except (struct.error, TypeError, OverflowError) as exc:
                    raise LayoutError(
                        f"属性 {name!r} 第 {vertex} 个顶点的分量 "
                        f"{component} 无法用 {code!r} 打包: {value!r}"
                    ) from exc
                position = base_offset + start + component * size
                out[position:position + size] = packed

    return bytes(out), vertex_count


def attr_size(fmt, count):
    """返回单个属性的字节数。

    参数：
        fmt: 格式名，可带归一化后缀 ``n``（如 ``"uint8n"``）。
        count: 分量个数，必须是 ``>= 1`` 的整数。

    契约：
        ``attr_size("float32", 1) == 4``；
        ``attr_size("float32", 3) == 12``（分量数必须乘进去）；
        ``attr_size("uint8n", 4) == 4``。

    异常：
        未知格式、``count`` 不是整数或 ``count < 1`` 时抛
        :class:`LayoutError`。
    """
    return _attr_size(fmt, count)


def stride(schema, align):
    """返回一个顶点占用的总字节数（已向上对齐到 ``align``）。

    参数：
        schema: 属性字典列表，按声明顺序排列。
        align: 正整数，对齐字节数。

    契约：
        ``raw = sum(attr_size(a["format"], a["count"]) for a in schema)``；
        返回 ``ceil(raw / align) * align``，因此结果一定 ``>= raw``
        且是 ``align`` 的整数倍。``raw == 0``（空 schema）时返回 ``0``。
        顶点末尾因对齐产生的空隙属于 stride，但不属于任何属性。

    异常：
        ``align`` 不是正整数（含 ``0``、负数）时抛 :class:`LayoutError`；
        未知格式或非法属性描述同样抛 :class:`LayoutError`。
    """
    return _stride(schema, align)


def offsets(schema):
    """返回每个属性在顶点内的起始字节偏移，顺序与 schema 一致。

    参数：
        schema: 属性字典列表。

    契约：
        结果为与 ``schema`` 等长的 ``list[int]``，按声明顺序从 ``0``
        开始累加 ``attr_size``；属性之间、属性内部都不插入对齐 padding
        （padding 只出现在 stride 末尾）。例如
        ``uint8``、``float32``×3、``uint16`` 三个属性的偏移是
        ``[0, 1, 13]``，而不是按格式大小排序后的结果。

    异常：
        schema 不是属性列表、未知格式、count 非正整数时抛
        :class:`LayoutError`。空 schema 返回 ``[]``。
    """
    return _offsets(schema)


def interleave(attrs, schema):
    """按 schema 把逐属性数组交错打包成一条顶点缓冲（小端）。

    参数：
        attrs: ``{属性名: 扁平序列}``；序列长度必须是该属性 ``count``
            的整数倍，所有属性推导出的顶点数必须一致。键集合必须与
            schema 的属性名集合完全相同（缺失或多余都算错误）。
        schema: 属性字典列表。

    契约：
        第 ``v`` 个顶点的第 ``k`` 个属性写入偏移
        ``v * stride(schema, DEFAULT_ALIGN) + offsets(schema)[k]``；
        数据一律使用 ``"<"`` 小端格式打包，数值按 ``count`` 个分量原样写入。
        返回的 ``bytes`` 长度严格等于
        ``stride(schema, DEFAULT_ALIGN) * 顶点数``，对齐补零字节为
        ``b"\\x00"``。空 schema 返回 ``b""``。

    异常：
        未知格式、属性缺字段/重名、键集合不符、长度不整除、
        各属性顶点数不一致、数值超出格式范围或类型不可打包时抛
        :class:`LayoutError`。
    """
    return _pack(attrs, schema, DEFAULT_ALIGN)[0]


def deinterleave(data, schema, count):
    """把 ``data`` 反向解包成 ``count`` 个顶点字典（小端）。

    参数：
        data: 顶点缓冲（bytes/bytearray）。
        schema: 属性字典列表。
        count: 顶点数，必须是 ``>= 0`` 的整数。

    契约：
        返回长度为 ``count`` 的 ``list[dict]``；``count == 1`` 的属性
        返回标量，``count > 1`` 的属性返回分量 ``tuple``，键名与 schema
        一致。``len(data)`` 必须严格等于
        ``stride(schema, DEFAULT_ALIGN) * count``，长度多一字节或少一
        字节都算错误（容差 0 字节），绝不读取下一个顶点的字节。
        空 schema 仅在 ``data == b""`` 时合法并返回 ``[]``。

    异常：
        长度不符、未知格式、非法 schema、``count`` 非法时抛
        :class:`LayoutError`。
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise LayoutError(f"data 必须是 bytes/bytearray: {type(data)!r}")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise LayoutError(f"count 必须是 >= 0 的整数: {count!r}")

    _require_schema(schema)
    if not schema:
        if len(data) != 0:
            raise LayoutError(
                f"空 schema 只接受空 data，实际长度 {len(data)}"
            )
        return []

    step = _stride(schema, DEFAULT_ALIGN)
    expected = step * count
    if len(data) != expected:
        raise LayoutError(
            f"data 长度 {len(data)} 与 stride {step} * count {count} "
            f"= {expected} 不符"
        )

    starts = _offsets(schema)
    raw = bytes(data)
    vertices = []
    for vertex in range(count):
        base_offset = vertex * step
        entry = {}
        for attr, start in zip(schema, starts):
            _base, code, size, _normalized = _parse_format(attr["format"])
            component_count = attr["count"]
            position = base_offset + start
            values = struct.unpack_from("<" + code * component_count, raw, position)
            entry[attr["name"]] = (
                values[0] if component_count == 1 else tuple(values)
            )
        vertices.append(entry)
    return vertices


def normalize_type(fmt):
    """返回 ``(基础格式名, 归一化除数)``。

    参数：
        fmt: 格式名，可带 ``n`` 后缀。

    契约：
        无 ``n`` 后缀的合法格式原样返回并配除数 ``1``，例如
        ``"float32" -> ("float32", 1)``、``"uint8" -> ("uint8", 1)``。
        有 ``n`` 后缀时：无符号整型除数为 ``2 ** bits - 1``，有符号整型
        除数为 ``2 ** (bits - 1) - 1``，因此 ``"uint8n" ->
        ("uint8", 255)``、``"int8n" -> ("int8", 127)``、
        ``"uint16n" -> ("uint16", 65535)``；最大值除以除数恰好为 ``1.0``。

    异常：
        未知格式、非整型加 ``n`` 后缀（如 ``"float32n"``）时抛
        :class:`LayoutError`。
    """
    base, _code, size, normalized = _parse_format(fmt)
    if not normalized:
        return base, 1
    bits = size * 8
    if base in _UNSIGNED_FORMATS:
        return base, (1 << bits) - 1
    return base, (1 << (bits - 1)) - 1


def validate(schema, data_len):
    """检查 schema 与 ``data_len`` 是否自洽，返回问题描述列表。

    参数：
        schema: 属性字典列表。
        data_len: 顶点缓冲的字节长度，非负整数。

    契约：
        合法时返回 ``[]``；发现问题时返回 ``list[str]``，每条字符串
        说明一个问题，不抛异常表示数据问题。必须覆盖：
        未知格式（含把整型格式拼错、浮点加 ``n``）、``count`` 不是
        正整数、属性缺字段或重名、``data_len`` 不是
        ``stride(schema, DEFAULT_ALIGN)`` 的整数倍（二次对齐/长度
        对不上），以及空 schema 时 ``data_len != 0``。
        未知格式绝不能静默当作 ``float32``。

    异常：
        仅当 ``schema`` 不是 list 或 ``data_len`` 不是 int 时抛
        :class:`LayoutError`（参数类型错误，不是数据问题）。
    """
    if not isinstance(schema, list):
        raise LayoutError(f"schema 必须是属性列表: {schema!r}")
    if not isinstance(data_len, int) or isinstance(data_len, bool):
        raise LayoutError(f"data_len 必须是整数: {data_len!r}")

    problems = _schema_problems(schema)

    if data_len < 0:
        problems.append(f"data_len 不能为负数: {data_len}")
        return problems

    if not schema:
        if data_len != 0:
            problems.append(f"空 schema 要求 data_len == 0，实际为 {data_len}")
        return problems

    # schema 有问题（缺字段、未知格式、count 非法等）时无法可靠推导
    # stride，问题已经收集到 problems 里，直接返回。
    try:
        step = _stride(schema, DEFAULT_ALIGN)
    except LayoutError:
        return problems

    if data_len % step != 0:
        problems.append(
            f"data_len {data_len} 不是 stride {step} 的整数倍"
        )

    return problems


def layout_batch(buffers, schema, config):
    """批量交错多个属性数组，返回一个结果字典。

    参数：
        buffers: 若干个 ``attrs`` 字典（契约同 :func:`interleave`）。
        schema: 所有 buffer 共用的属性字典列表。
        config: 配置字典，支持 ``align``（正整数，缺省
            ``DEFAULT_ALIGN``）。

    契约：
        每个 buffer 独立打包成 ``stride(schema, align) * 顶点数`` 字节，
        再按 buffers 的顺序拼接。返回::

            {"data": bytes,       # 全部批次的拼接结果
             "offsets": list[int],# 每个批次在 data 中的起始字节
             "counts": list[int], # 每个批次的顶点数
             "stride": int}

        ``offsets`` 严格递增，``offsets[i] == sum(stride * counts[:i])``；
        批次顺序保持不变。各批次必须使用彼此独立的临时缓冲，禁止共用
        同一个可变输出缓冲，否则批次之间会串数据。
        任一 buffer 非法（未知格式、长度不整除、顶点数不一致、键集合
        不符、数值越界等）或 ``align`` 非法时抛 :class:`LayoutError`：
        整个调用失败，不返回部分结果，也不修改调用方传入的 buffers/
        config（整体回滚）。空 ``buffers`` 返回空的 data/offsets/counts。

    异常：
        输入类型不符或上述任一校验失败时抛 :class:`LayoutError`。
    """
    if not isinstance(buffers, list):
        raise LayoutError(f"buffers 必须是列表: {buffers!r}")
    if not isinstance(config, dict):
        raise LayoutError(f"config 必须是字典: {config!r}")

    align = _require_align(config.get("align", DEFAULT_ALIGN))

    # 先整体打包到彼此独立的局部缓冲；任一批次失败都不会留下部分结果，
    # 也不会触碰调用方传入的 buffers/config。
    chunks = []
    counts = []
    starts = []
    cursor = 0
    step = _stride(schema, align)
    for index, attrs in enumerate(buffers):
        try:
            chunk, vertex_count = _pack(attrs, schema, align)
        except LayoutError as exc:
            raise LayoutError(f"第 {index} 个 buffer 非法: {exc}") from exc
        starts.append(cursor)
        counts.append(vertex_count)
        chunks.append(chunk)
        cursor += len(chunk)

    return {
        "data": b"".join(chunks),
        "offsets": starts,
        "counts": counts,
        "stride": step,
    }
