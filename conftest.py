"""pytest 启动入口：把仓库根目录加入 ``sys.path``。

这样从任意工作目录执行 ``python -m pytest tests -q`` 时，
``import vlayout`` 都能命中仓库内的源码包。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
