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


class LayoutError(ValueError):
    """顶点布局或顶点缓冲数据不合法。

    ``attr_size``/``stride``/``interleave``/``deinterleave``/
    ``normalize_type``/``layout_batch`` 在输入违反契约时抛出本异常；
    ``validate`` 则不抛异常，而是把问题字符串收集到返回值里。
    """


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
