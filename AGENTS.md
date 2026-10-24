# AGENTS.md

面向维护者的项目约定。用户文档见 `README.md` / `README_CN.md`。

## 身份约定

- PyPI 发行名：`TextGrocery_Shop`，pip 规范化为 `textgrocery-shop`。
- Python 导入名：`tgrocery`（不要改包目录名，旧代码依赖它）。
- 安装命令统一写 `pip install textgrocery-shop`，代码示例统一写
  `from tgrocery import Grocery`。
- 版本号两处必须一致：`pyproject.toml` 的 `[project].version` 与
  `tgrocery/__init__.py` 的 `__version__`。

## 许可证

`pyproject.toml`、`README.md`、`LICENSE.txt` 统一为 BSD-3-Clause。
注意：上游 `2shou/TextGrocery` 仓库的 LICENSE.txt 是 GPLv3；若以后直接复制
上游代码，必须重新评估许可证兼容性。

## 常用命令

```bash
python3 -m pytest -q
ruff check .
ruff format --check .
rm -rf build dist *.egg-info
python3 -m build
python3 -m twine check dist/*
```

构建要求 `setuptools>=77`（PEP 639 的 `license` / `license-files`）。
系统自带旧版 setuptools 时不要用 `python3 -m build --no-isolation`。

## 发布

首次发布前确认 <https://pypi.org/project/textgrocery-shop/> 仍可用。
完整流程（TestPyPI、twine 环境变量、Trusted Publishing）见两份 README 的
“打包与发布 / Packaging and publishing”章节。
