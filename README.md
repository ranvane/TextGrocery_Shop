# TextGrocery_Shop

一个简单、跨平台的中文短文本分类工具。支持 **Python 3.10+**，安装时**不需要任何 C 编译器**，
并完整保留旧版 `Grocery` 的调用方式。

内置 [结巴分词](https://github.com/fxsjy/jieba)，在词特征之外增加字符 n-gram、TF-IDF 加权与
L2 正则逻辑回归，对新闻标题、商品标题、工单摘要这类超短文本更友好。

其他语言：[English](README_EN.md)

## 项目来历

**原版 TextGrocery**（作者 [2shou](https://github.com/2shou/TextGrocery)）诞生于 Python 2 时代，
是中文短文本分类里常用的轻量基线。它的技术路线是：

- 用 jieba 做中文分词；
- 用词 unigram / bigram 计数构造特征，再做二值化和 L2 归一化；
- 通过 `ctypes` 调用编译好的 **LibShortText / LibLinear** C++ 动态库完成训练与预测；
- 模型用 `pickle` 保存。

这套方案在当时非常简单直接，但也积累了几个问题：

1. **Python 2 已停止维护**，原代码无法在 Python 3.10+ 上运行；
2. **安装依赖 C 编译器**，Windows 上需要 MSVC、Linux/macOS 上需要 `make + gcc`，
   而且编译出的 `.so` 不能跨平台复用；
3. **模型不可移植**：`pickle` 与 Python 版本、NumPy 版本绑定，换环境经常直接失败；
4. **超短文本效果有限**：十几字的标题分词后只剩几个词，未登录词直接丢失，特征向量接近全零；
5. **只有决策值，没有概率**，也不方便扩展特征。

**0.2.x 的目标**是保留“简单好用”的 API，同时把上述问题一次性解决：用纯 Python + NumPy
重写训练器，去掉编译依赖；加入字符 n-gram 与 TF-IDF；模型改用 JSON + NumPy 数组；
并补上概率输出、分词模式、用户词典等实用选项。旧版的设计思路仍然值得借鉴，
感谢原作者与 LibShortText / LibLinear 的工作。

## 新版特性

- **零编译安装**：纯 Python + NumPy，`pip install` 在 Windows / Linux / macOS 行为一致。
- **API 不变**：`Grocery(name)` → `train()` → `save()` → `load()` → `predict()` / `test()`。
- **更适合短文本**：字符 2–3gram + sublinear TF + 平滑 IDF + L2 归一化。
- **输出概率**：`result.probabilities`，同时保留原始决策值 `result.dec_values`。
- **模型可搬运**：`metadata.json` + `vocab.json` + `model.npz`，不使用 `pickle`。
- **分词可配置**：全模式 / 精确模式 / 搜索引擎模式，支持用户词典、停用词、NFKC 归一化。
- **速度**：10 类新闻标题模型上约 **80–85 µs/条**，训练全程向量化。

## 安装

```bash
pip install textgrocery-shop
```

PyPI 发行名为 `TextGrocery_Shop`（会被规范化为 `textgrocery-shop`）；导入名仍是
`tgrocery`，老代码不需要改。

从源码安装：

```bash
pip install .
```

依赖：`jieba`、`numpy`；Python ≥ 3.10。

## 快速开始

```python
from tgrocery import Grocery

# 创建一个杂货铺，名字同时作为模型目录名
grocery = Grocery("sample")

# 用列表训练：[(标签, 文本), ...]
train_src = [
    ("education", "名师指导托福语法技巧：名词的复数形式"),
    ("education", "中国高考成绩海外认可 是狼来了吗"),
    ("sports", "图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼"),
    ("sports", "四川丹棱举行全国长距登山挑战赛 近万人参与"),
]
grocery.train(train_src)

# 也可以从文件训练：标签<TAB>文本（UTF-8）
# grocery.train("train_ch.txt")

grocery.save()

# 加载模型（名字与保存时一致）
new_grocery = Grocery("sample")
new_grocery.load()

# 预测
result = new_grocery.predict("考生必读：新托福写作考试评分标准")
print(result)  # education（与旧版一样直接打印标签）
print(result.dec_values)  # {'education': 2.13, 'sports': -2.05, ...}
print(result.probabilities)  # {'education': 0.91, 'sports': 0.06, ...}

# 批量测试
test_src = [
    ("education", "福建春季公务员考试报名18日截止 2月6日考试"),
    ("sports", "意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜"),
]
test_result = new_grocery.test(test_src)
print(test_result.accuracy_overall)
test_result.show_result()  # 逐标签精确率 / 召回率表格
```

更多示例见 [sample/](sample/)。

## 为什么对短文本更有效

原版的特征是“词计数 → 二值化 → L2”。标题再短一些时，分词后往往只剩 3–5 个词，
大量词不在词表里，最终输入向量接近全零。新版从三个层面缓解：

### 1. 字符 n-gram 兜底

```text
"中国高考" → c:中国, c:国高, c:高考, c:中国高, c:国高考
```

即使“高考生”“考研”是未登录词，只要共享“高考”两个字符，就能命中 `c:高考`。
词特征和字符特征分别加 `w:` / `c:` 前缀，互不覆盖。

### 2. TF-IDF 加权

- **sublinear TF**：`1 + log(tf)`，避免一个词重复出现就主导整条文本；
- **平滑 IDF**：`log((1 + N) / (1 + df)) + 1`，降低“的、是”等高频词权重，且不会出现 0/负值；
- **L2 行归一化**：让长短文本处于同一尺度。

### 3. 分类器与噪声控制

- OvR 逻辑回归 + L2 正则，用 **L-BFGS**（Armijo 回溯线搜索）优化；
- `min_df` 自动策略：语料 ≥ 1000 篇时过滤只出现一次的特征，小语料则全部保留；
- `class_weight="balanced"` 处理类别不平衡；
- 稀疏矩阵乘法用 `numpy.bincount`，计算仍发生在 C 层。

目标函数：

$$
\min_{w_k,b_k}\;\frac{1}{2}\lVert w_k\rVert^2
+ C\sum_{i=1}^{n}s_i\log\!\left(1+\exp\!\left(-y_{ik}(w_k^\top x_i+b_k)\right)\right)
$$

预测时取 `argmax_k (w_k^T x + b_k)`，再经 softmax 得到概率。

## 分词与文本预处理

默认分词器仍是 jieba 全模式（`cut_all=True`），这是原版行为，也是当前数据上效果最好的模式。
新增选项都通过**仅关键字参数**提供，不传时行为不变。

```python
grocery = Grocery(
    "sample",
    tokenizer="full",  # full | precise | search
    normalize="none",  # none | nfkc
    char_ngram_mode="global",  # global | word_boundary
    mask_patterns=False,  # URL / 邮箱 / 数字占位符
    stopwords=["的", "了"],  # 也可传文件路径
    user_dict="domain_words.txt",  # 每行一个领域词
)
```

### 分词模式

| 模式 | 说明 | 实测准确率 |
|---|---|---:|
| `full`（默认） | 全模式，输出所有可能成词的片段，召回高 | **90.22%** |
| `precise` | 精确模式，只输出一条最可能的切分 | 89.28% |
| `search` | 搜索引擎模式，精确模式 + 长词子词 | 89.40% |

同一份 THUCNews 30k/5k 数据、相同特征与超参数下，全模式仍最好。精确模式特征更少
（149k vs 164k）、训练略快，适合对内存/延迟更敏感的场景；领域不同可以自行 A/B。

### 用户词典

领域实体、产品名、缩写在通用词典里经常被切碎。把词表文件传给 `user_dict`，
初始化时会注册进 jieba，并**随模型一起写入 metadata**，换机器加载也能复现：

```python
grocery = Grocery("sample", user_dict="domain_words.txt")
# 或直接传列表
grocery = Grocery("sample", user_dict=["烯晶算力网络", "星云一体机"])
```

当前 `user_dict` 只接受 **UTF-8 文本、每行一个词**；词内可以包含空格，但不能带频次、
拼音或词性列。搜狗导出的 `.scel` 是二进制细胞词库，**不能直接传给 `user_dict`**，
需要先转换成纯文本。下面是一套完整的转换与清洗流程。

#### 1. 把 `.scel` 转成文本

使用[深蓝词库转换](https://github.com/studyzy/imewlconverter)（ImeWlConverter）、`scel2txt` 等工具，把 `.scel` 导出为 `.txt`。
不同工具的输出列不完全一样，常见形式有：

```text
烯晶算力网络
烯晶算力网络	10
烯晶算力网络	xijing suanli wangluo	10
烯晶算力网络 10 n
烯晶算力网络,10,名
```

前四种分别对应“纯词”“词 + 频次”“词 + 拼音 + 频次”“jieba 风格词 + 频次 + 词性”。
我们的加载器只认第一种，其他格式需要清洗。**建议在转换工具里把输出列配置为
TAB 分隔**（`词<TAB>频次` 或 `词<TAB>拼音<TAB>频次`）；如果工具只能输出空格分隔的
`词 拼音 频次`，请先改成 TAB/CSV 再清洗，否则拼音会被当成词的一部分。

#### 2. 清洗成“每行一个词”

把下面的脚本保存为 `sogou_to_userdict.py`，按实际情况修改输入输出路径后运行：

```python
from pathlib import Path
import re


def sogou_to_userdict(src: str, dst: str) -> None:
    """把搜狗转换后的词表清洗成 TextGrocery_Shop 的 user_dict 格式。

    规则：
    1. 优先按 TAB / 逗号取第一列（词），丢掉频次、拼音、词性；
    2. 没有分隔符时，去掉行尾的“频次 [词性]”，兼容 "词 10 n"；
    3. 跳过空行与 # 注释，按原顺序去重；
    4. 统一输出 UTF-8（无 BOM），每行一个词。
    """
    words: list[str] = []
    seen: set[str] = set()

    for raw in Path(src).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # TSV / CSV：第一列是词，其余列是频次、拼音或词性
        first = re.split(r"[\t,]", line, maxsplit=1)[0].strip()

        # 空格分隔：只处理 "词 频次 [词性]"；
        # 若是 "词 拼音 频次"，请在转换工具中改用 TAB/CSV 输出。
        # 注意：没有 TAB/逗号时，"iPhone 15" 这类词尾数字会被误删；
        # 如果词表含此类产品名，请优先使用 TAB 分隔的导出格式。
        word = re.sub(r"\s+\d+(?:\s+[a-z]+)?$", "", first).strip()

        if not word or word in seen:
            continue
        seen.add(word)
        words.append(word)

    Path(dst).write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"写入 {len(words)} 个词 -> {dst}")


if __name__ == "__main__":
    sogou_to_userdict("sogou_export.txt", "domain_words.txt")
```

运行后检查一下结果：

```bash
python -c "from pathlib import Path; w=Path('domain_words.txt').read_text(encoding='utf-8').splitlines(); print(len(w)); print(w[:20])"
```

#### 3. 交给 `user_dict`

```python
from tgrocery import Grocery

grocery = Grocery(
    "sample",
    tokenizer="precise",  # 领域词较多时，精确模式通常更稳
    user_dict="domain_words.txt",
)
print(len(grocery.feature_config.user_words))  # 已加载的词数

grocery.train(train_src)

# 验证领域词确实进入了词表（词特征带 w: 前缀）
print("w:烯晶算力网络" in grocery.model.text_converter.feature2idx)
```

保存后，词表会写入 `metadata.json` 的 `features.user_words`：

```json
{
  "features": {
    "tokenizer": "precise",
    "user_words": ["烯晶算力网络", "星云一体机"]
  }
}
```

之后 `Grocery("sample").load()` 会从 metadata 重新注册这些词，**换机器加载模型时
不需要再带原始词典文件**。

#### 4. 注意事项

- **不要直接导入全量搜狗词库**。通用词对分类帮助很小，却会让 `metadata.json`
  膨胀、`load()` 变慢。建议只保留领域实体、产品名、型号、缩写、人名地名，
  规模控制在几千到一两万词。
- 搜狗词库通常带频次。如果转换工具保留了频次，可以先按频次阈值过滤，再做上面的清洗。
- 开启 `lowercase=True`（默认）时，英文词会被转成小写后写入 metadata；
  如果产品名大小写有意义，请设置 `lowercase=False`。
- 同一进程内 jieba 的词典是全局状态：多个 `Grocery` 使用不同 `user_dict` 时会互相影响。
  需要严格隔离时，把不同模型放到不同进程中使用。
- 不同 jieba 版本的基础词典可能略有差异；对分词结果要求完全可复现时，请固定 jieba 版本。
- 搜狗词库有各自的许可条款；把词表写入模型 metadata 意味着它会随模型一起分发，
  请确认授权范围。
- `user_dict` 只影响 jieba 分词；使用 `custom_tokenize` 时不会生效。

### 停用词

`stopwords` 支持文件路径或可迭代对象，只作用于词特征与 `word_boundary` 字符特征。
`global` 字符特征不会删除停用词，否则会破坏相邻字符 n-gram。

### 归一化与占位符

- `normalize="nfkc"`：全角字母/数字/标点转半角，统一 `ＡＢＣ` 与 `ABC`；
- `mask_patterns=True`：把 URL、邮箱、数字替换为 `__url__` / `__email__` / `__num__`，
  适合链接、价格、日期很多 UGC 文本，避免特征爆炸。

### 字符 n-gram 模式

- `global`（默认）：去掉空白后整串滑窗，中文标题效果最好；
- `word_boundary`：只在词内部滑窗并补边界空格（类似 sklearn 的 `char_wb`），
  避免英文/中英混排生成跨词噪声。分词更重时开销会上升，建议搭配 `tokenizer="precise"`。

### 效率

默认路径（全模式 + 全局字符 n-gram + 不做 NFKC/占位符替换）与之前完全一致：
30k 训练约 **26.7 s**，5k 预测约 **0.42 s**。新增的归一化、停用词、用户词典
只在开启时产生额外开销；`word_boundary` 是唯一可能明显变慢的选项。

## 实测效果

数据：THUCNews 10 类新闻标题（训练 30,000 / 测试 5,000，标题中位数 19 字）。
三组配置使用相同的分类器与超参数（`C=8`、`max_iter=50`、`min_df=2`），只改变特征：

| 配置 | 完整标题 | 截断到 10 字 | 训练耗时 |
|---|---:|---:|---:|
| `legacy`（词 1–2gram、二值、无 IDF） | 88.82% | 82.86% | 11.0 s |
| `word_tfidf`（词 1–2gram + TF-IDF） | 89.20% | 83.74% | 10.8 s |
| **`word_char`（词 + 字符 2–3gram + TF-IDF）** | **90.22%** | **84.66%** | 26.7 s |

预测：5,000 条约 0.42 秒（约 84 微秒/条）。

> 该数据与旧版 README 中的 48k/16k、32 类基准不是同一份数据，绝对数值不可直接比较。
> 请运行 `benchmark.py` 在你自己的数据上验证。

## 基准测试

```bash
# 对比 legacy / word_tfidf / word_char
python benchmark.py --train cnews.train.txt --test cnews.test.txt

# 模拟超短文本（每条截断到前 10 个字符）
python benchmark.py --train cnews.train.txt --test cnews.test.txt --truncate 10

# 限制样本量，快速试验
python benchmark.py --train cnews.train.txt --test cnews.test.txt \
    --train-size 30000 --test-size 5000
```

数据格式默认是 `文本<TAB>标签`；若是 `标签<TAB>文本`，加 `--label-column first`。

## 参数说明

所有新增参数都是**仅关键字参数**，旧代码无需修改。

```python
Grocery(
    "name",
    custom_tokenize=None,  # 自定义分词函数，优先级高于 tokenizer
    C=8.0,  # L2 正则强度的倒数
    class_weight=None,  # None | "balanced" | {"标签": 权重}
    word_ngrams=(1, 2),  # 词级 n-gram
    char_ngrams=(2, 3),  # 字符级 n-gram；() 表示关闭
    lowercase=True,
    min_df=None,  # None = 自动（<1000 篇取 1，否则取 2）
    max_features=None,
    binary=False,  # True 时退化为旧版二值特征
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True,
    norm="l2",
    tokenizer="full",  # full | precise | search
    normalize="none",  # none | nfkc
    char_ngram_mode="global",  # global | word_boundary
    mask_patterns=False,
    stopwords=None,
    user_dict=None,
    max_iter=50,
    tol=1e-4,
    verbose=False,
)
```

调参建议：

- 数据量大、内存吃紧：`max_features=300000` 或提高 `min_df`；
- 类别不平衡：`class_weight="balanced"`；
- 追求速度：`char_ngrams=()`，只用词特征 + TF-IDF；
- 英文/中英混排较多：`tokenizer="precise", char_ngram_mode="word_boundary"`；
- 全角字符或 UGC 文本：`normalize="nfkc"`，必要时加 `mask_patterns=True`；
- 完全复现旧版特征：`word_ngrams=(1, 2), char_ngrams=(), binary=True,
  sublinear_tf=False, use_idf=False, normalize="none"`。

## 模型目录格式

`save()` 生成一个自包含、可移植的目录：

```text
name/
    id                 # 模型哈希
    metadata.json      # 标签、特征配置（含分词/停用词/用户词典）、分类器参数
    vocab.json         # 特征词表（UTF-8）
    vectorizer.npz     # IDF 向量
    model.npz          # 权重矩阵 + 偏置
```

不使用 `pickle`，模型可在不同 Python 版本、不同操作系统之间搬运；
出问题时也可以直接打开 `metadata.json` / `vocab.json` 检查。

## 兼容性与迁移

- **调用方式不变**：`Grocery(name)`、`train()`、`save()`、`load()`、`predict()`、
  `test()`，以及 `GroceryPredictResult` / `GroceryTestResult` 的字段与旧版一致。
- `predict()` 返回对象可以 `print` 出标签；新增 `probabilities`。
- 旧的 C 后端（LibShortText / LibLinear）已移除，当前只保留纯 NumPy 实现。
- **0.1.x 旧模型不可加载**：旧模型依赖 pickle 与 C 后端，请用新版重新训练一次。
- 0.2.0 早期模型仍可加载：缺失的新配置会自动回退到旧行为。

## 发布到 PyPI

仓库使用 GitHub Actions + PyPI Trusted Publishing 自动发布：在 GitHub 发布
Release 时，`.github/workflows/pypi_publish.yml` 自动构建 wheel/sdist 并上传
PyPI。整个流程不需要在仓库或本地保存 PyPI API token。

### 一次性配置

1. 发行名固定为 `TextGrocery_Shop`（pip 安装名 `textgrocery-shop`）。在 PyPI 的
   **Publishing** 页面添加 GitHub publisher：owner 填 `ranvane`、repository 填
   `TextGrocery_Shop`、workflow 填 `pypi_publish.yml`、environment 填 `pypi`；
   项目尚不存在时使用 **pending publisher**。
2. 在 GitHub 仓库 **Settings → Environments** 创建 `pypi` 环境，建议添加
   required reviewers。
3. 确认 `pyproject.toml` 的 `[project].version` 与
   `tgrocery/__init__.py` 的 `__version__` 一致。

### 发布流程

```bash
# 1. 提交代码，等 CI 通过后打 tag
git tag v0.2.0
git push origin v0.2.0
```

2. 打开 GitHub 仓库的 **Releases → Draft a new release**，选择 tag `v0.2.0`，
   标题填 `v0.2.0`，点击 **Publish release**。
3. 工作流 `Upload Python Package` 会自动执行：
   - `python -m build` 生成 `textgrocery_shop-0.2.0-py3-none-any.whl` 和
     `textgrocery_shop-0.2.0.tar.gz`；
   - 通过 `pypa/gh-action-pypi-publish@release/v1` 上传到 PyPI。
4. 在 **Actions** 页面确认发布 job 成功。

工作流中的 `permissions: id-token: write` 必须保留，否则 OIDC 无法换取 PyPI
短期凭证。

### 发布后验证

```bash
pip install --no-cache-dir textgrocery-shop==0.2.0
python -c "import tgrocery; print(tgrocery.__version__)"
```

PyPI 版本号和文件名一旦上传就不能复用；再次发布前必须提升版本号，并创建新的
tag 与 Release。

### 发布检查清单

- [ ] `pyproject.toml` 与 `tgrocery/__init__.py` 的版本号一致
- [ ] `ruff check .`、`ruff format --check .`、`pytest` 全部通过
- [ ] 代码已提交并推送，CI 通过
- [ ] tag（如 `v0.2.0`）已推送
- [ ] 已在 GitHub 发布 Release
- [ ] `Upload Python Package` 工作流成功
- [ ] 正式 PyPI 可用 `pip install textgrocery-shop==0.2.0` 安装
- [ ] 未提交 `dist/`、`.pypirc`、token 或本地模型

## 开发

```bash
pip install -e .[test]
pytest
ruff check .
ruff format --check .
```

## 更新日志

- **2026-10-24**：PyPI 发行名改为 `TextGrocery_Shop`
  （`pip install textgrocery-shop`）；导入名仍为 `tgrocery`。
  许可证元数据与 `LICENSE.txt` 统一为 BSD-3-Clause。
- **2026-10-24**：发布流程改为 GitHub Release + Trusted Publishing，工作流
  `.github/workflows/pypi_publish.yml` 自动上传 PyPI。

## 许可证

BSD，见 [LICENSE.txt](LICENSE.txt)。
