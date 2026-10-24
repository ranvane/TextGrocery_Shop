"""兼容性构建入口。

项目元数据（名称、版本、依赖、Python 版本要求）全部定义在
``pyproject.toml`` 中，本文件只保留一个最小的 ``setup()`` 调用，
以便老脚本仍可使用 ``python setup.py ...`` 之类的旧式命令。

注意：新版本不再编译任何 C 扩展，``pip install .`` 在
Windows / Linux / macOS 上都不需要编译器。
"""

from setuptools import setup

setup()
