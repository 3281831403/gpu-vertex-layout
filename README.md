# gpu-vertex-layout

一个只依赖 Python 标准库与 pytest 的顶点属性交错布局（vertex attribute
interleaving）小工具。8 个入口目前全部是 `NotImplementedError` 桩，配套 15 条
验收用例初始全红，属于 Bugfix 题。

## 题面

把顶点属性改成交错布局后，模型整体错位、颜色还全反了：attr_size 只算单个分量、完全忽略 count；stride 不把总字节数向上取整到 align，最后一个属性越界写出 stride；offsets 按格式大小排序而不是按声明顺序排列，属性整体错位；interleave 用大端写缓冲，GPU 按小端解释后数值全错；deinterleave 不校验 data 长度，会把下一个顶点的字节读进来；归一化除数写成 256，uint8 的最大值除以它到不了 1；未知格式被静默当成 float32；validate 不检查 stride 是否整除 data 长度，二次对齐后长度对不上；layout_batch 让各批次共用同一个输出缓冲，批次之间的数据互相串；stride 不校验 align=0，直接除零崩溃。vlayout/interleave.py 是核心，tests/test_interleave.py 15个红。要修的是：attr_size 按 count 计算字节数、stride 向上对齐且 align 必须为正整数、offsets 保持声明顺序、统一小端打包、解包前严格校验长度、归一化使用 2^n-1、未知格式抛 LayoutError、validate 报出 stride 不整除、批量使用独立缓冲且任一缓冲非法时整体回滚、空 schema 返回空布局。

## 运行方式

```powershell
cd D:\gsb-20261119\gpu-vertex-layout
python -m pytest tests -q
```

初始（未修复）状态下命令必须恰好以 `15 failed` 结束，且没有 collection
error、import error 或 syntax error；示例输出尾部：

```text
FAILED tests/test_interleave.py::test_defect_01_attr_size_counts_components
FAILED tests/test_interleave.py::test_defect_02_stride_rounds_up_to_align
FAILED tests/test_interleave.py::test_defect_03_offsets_follow_declaration_order
FAILED tests/test_interleave.py::test_defect_04_interleave_is_little_endian
FAILED tests/test_interleave.py::test_defect_05_deinterleave_checks_length
FAILED tests/test_interleave.py::test_defect_06_normalize_divisor_is_max_integer
FAILED tests/test_interleave.py::test_defect_07_unknown_format_is_rejected
FAILED tests/test_interleave.py::test_defect_08_validate_reports_stride_mismatch
FAILED tests/test_interleave.py::test_defect_09_batch_buffers_do_not_mix
FAILED tests/test_interleave.py::test_defect_10_batch_rejects_zero_align
FAILED tests/test_interleave.py::test_attr_size_single_float32_is_four_bytes
FAILED tests/test_interleave.py::test_stride_rejects_zero_align
FAILED tests/test_interleave.py::test_empty_schema_yields_empty_layout
FAILED tests/test_interleave.py::test_interleave_deinterleave_roundtrip
FAILED tests/test_interleave.py::test_batch_all_success_preserves_order
15 failed in 0.08s
```

修复完成后应变为 `15 passed`。全程离线、无随机、无外部依赖。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 接口契约

属性统一用 `{"name": str, "format": str, "count": int}` 描述；`schema` 是
按声明顺序排列的属性列表，`align` 是正整数对齐字节数，模块默认
`DEFAULT_ALIGN = 4`。

支持的 `format`（字节数）：

| 字节数 | 格式名 |
| --- | --- |
| 1 | `int8`, `uint8` |
| 2 | `int16`, `uint16`, `float16` |
| 4 | `int32`, `uint32`, `float32` |
| 8 | `int64`, `uint64`, `float64` |

格式名末尾加 `n` 表示归一化整型（如 `uint8n`、`int16n`），只允许整型；
缓冲区内存原始整数，`normalize_type` 返回到 `[0, 1]` / `[-1, 1]` 的除数。
所有打包/解包一律小端（`"<"`），长度容差 0 字节。

| 入口 | 契约摘要 |
| --- | --- |
| `attr_size(fmt, count)` | 单属性字节数 = 分量大小 × count |
| `stride(schema, align)` | 全部属性字节数之和向上对齐到 align 的整数倍 |
| `offsets(schema)` | 各属性起始偏移，严格按声明顺序 |
| `interleave(attrs, schema)` | 交错打包，长度 = stride × 顶点数，小端 |
| `deinterleave(data, schema, count)` | 反向解包，`count == 1` 返回标量，否则返回 tuple |
| `normalize_type(fmt)` | `(基础格式, 归一化除数)`，如 `uint8n -> ("uint8", 255)` |
| `validate(schema, data_len)` | 返回问题字符串列表，合法时 `[]` |
| `layout_batch(buffers, schema, config)` | 批量交错，返回 `data/offsets/counts/stride`，失败整体回滚 |

## tests 不可修改

`tests/test_interleave.py` 是评测标准，15 条用例覆盖 10 个缺陷与 5 条组合
场景，不允许修改；`conftest.py` 只负责把仓库根加入 `sys.path`。
