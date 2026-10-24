# TextGrocery_Shop

A simple, cross-platform short-text classification tool for **Python 3.10+**.
It installs with **no C compiler**, and keeps the original `Grocery` API unchanged.

[结巴/jieba](https://github.com/fxsjy/jieba) is the default tokenizer. On top of the
original word-count baseline, 0.2.x adds character n-grams, TF-IDF weighting and
L2-regularized logistic regression, which helps on very short texts such as news
titles, product titles and ticket summaries.

Other languages: [详细中文文档](README.md)

## Origin and motivation

The **original TextGrocery** by [2shou](https://github.com/2shou/TextGrocery) was a
popular lightweight Chinese short-text baseline from the Python 2 era. It:

- tokenized Chinese with jieba;
- built word unigram/bigram counts, then binarized and L2-normalized them;
- called a compiled **LibShortText / LibLinear** C++ library through `ctypes`;
- stored models with `pickle`.

That design was simple and fast, but it accumulated real problems:

1. Python 2 reached end of life, so the code no longer runs on Python 3.10+;
2. installation required a C toolchain (MSVC on Windows, `make + gcc` elsewhere),
   and the resulting `.so` files were not portable;
3. `pickle` models were tied to specific Python/NumPy versions;
4. ultra-short texts produced extremely sparse word features and lost unseen words;
5. there were no probability outputs and little room for feature extensions.

**0.2.x** keeps the simple API and rewrites the internals with pure Python + NumPy:
no compilation, character n-grams, TF-IDF, portable model files, probabilities,
and optional tokenizer/normalization controls. Thanks to the original author and
to the LibShortText / LibLinear projects for the design inspiration.

## Highlights

- **Zero-compilation install**: pure Python + NumPy on Windows, Linux and macOS.
- **API unchanged**: `Grocery(name)` → `train()` → `save()` → `load()` → `predict()` / `test()`.
- **Better on short text**: character 2–3 grams + sublinear TF + smooth IDF + L2.
- **Probabilities**: `result.probabilities` alongside raw `result.dec_values`.
- **Portable models**: `metadata.json` + `vocab.json` + `model.npz`, no `pickle`.
- **Configurable tokenization**: full / precise / search modes, user dictionary,
  stopwords and NFKC normalization.
- **Fast**: about **80–85 µs per prediction** on a 10-class news-title model.

## Install

```bash
pip install textgrocery-shop
```

The PyPI distribution is `TextGrocery_Shop` (normalized to `textgrocery-shop`);
the import name stays `tgrocery`, so existing code needs no changes.

From source:

```bash
pip install .
```

Dependencies: `jieba`, `numpy`. Python ≥ 3.10.

## Quick start

```python
from tgrocery import Grocery

grocery = Grocery("sample")

train_src = [
    ("education", "Student debt to cost Britain billions within decades"),
    ("education", "Chinese education for TV experiment"),
    ("sports", "Middle East and Asia boost investment in top level sports"),
    ("sports", "Summit Series look launches HBO Canada sports doc series: Mudhar"),
]
grocery.train(train_src)

# Or train from a file: label<TAB>text (UTF-8)
# grocery.train("train_ch.txt")

grocery.save()

new_grocery = Grocery("sample")
new_grocery.load()

result = new_grocery.predict("Abbott government spends $8 million on higher education media blitz")
print(result)  # education
print(result.dec_values)  # {'education': 2.13, 'sports': -2.05}
print(result.probabilities)  # {'education': 0.98, 'sports': 0.02}

test_result = new_grocery.test(train_src)
print(test_result.accuracy_overall)
test_result.show_result()  # per-label precision / recall table
```

More examples live in [sample/](sample/).

## Why it works better on short text

The original pipeline was “word counts → binary → L2”. On titles of a dozen
characters, segmentation leaves only a few words and many of them are out of
vocabulary, so the feature vector is almost all zeros. 0.2.x mitigates this in
three ways:

1. **Character 2–3 grams** (`c:中国`, `c:国高`, `c:高考`, …) link unseen words
   through shared characters without depending on segmentation.
2. **TF-IDF**: sublinear TF (`1 + log tf`), smooth IDF (`log((1+N)/(1+df)) + 1`)
   and L2 normalization keep repeated words and document length from dominating.
3. **Better classifier and noise control**: one-vs-rest logistic regression with
   L2 regularization, optimized by L-BFGS with Armijo line search; automatic
   `min_df` (1 below 1000 documents, 2 above); optional balanced class weights.
   Sparse products use `numpy.bincount`, so the heavy lifting stays in C.

## Tokenization and text preprocessing

The default remains jieba **full mode** (`cut_all=True`), matching the original
behavior and performing best on our benchmark. All new options are keyword-only;
omitting them preserves the old behavior.

```python
grocery = Grocery(
    "sample",
    tokenizer="full",  # full | precise | search
    normalize="none",  # none | nfkc
    char_ngram_mode="global",  # global | word_boundary
    mask_patterns=False,  # mask URLs / emails / numbers
    stopwords=["的", "了"],  # iterable or file path
    user_dict="domain_words.txt",  # one word per line
)
```

### Tokenizer modes

| Mode | Description | Accuracy |
|---|---|---:|
| `full` (default) | all possible words; high recall | **90.22%** |
| `precise` | single most likely segmentation | 89.28% |
| `search` | precise + sub-words of long words | 89.40% |

Measured on THUCNews 30k/5k with identical features and hyper-parameters.
Full mode still wins; precise mode produces fewer features (149k vs 164k) and
trains slightly faster. Benchmark on your own domain before switching.

### User dictionary

Domain entities and product names are often split incorrectly. Pass a word file
(or an iterable) as `user_dict`; words are registered with jieba and **saved into
the model metadata**, so loading on another machine reproduces the same features.

```python
grocery = Grocery("sample", user_dict="domain_words.txt")
grocery = Grocery("sample", user_dict=["烯晶算力网络", "星云一体机"])
```

`user_dict` accepts UTF-8 text with **one word per line**. Sogou's exported `.scel`
cell dictionaries are binary and cannot be passed directly; convert them first.

#### Using a Sogou thesaurus as `user_dict`

1. Convert the `.scel` file to `.txt` with a tool such as
   [ImeWlConverter / 深蓝词库转换](https://github.com/studyzy/imewlconverter) or
   `scel2txt`.
2. Keep only the word column. Exports often contain frequency, pinyin or POS
   columns (`word<TAB>freq`, `word pinyin freq`, `word 10 n`, `word,freq,pos`).
   Prefer a **TAB-separated** export format; a space-separated
   `word pinyin freq` line cannot be split reliably without knowing where the
   word ends.
3. Save as **UTF-8 without BOM**, one word per line.
4. Pass the cleaned file to `user_dict`.

Cleaning script:

```python
from pathlib import Path
import re


def sogou_to_userdict(src: str, dst: str) -> None:
    words: list[str] = []
    seen: set[str] = set()

    for raw in Path(src).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # TSV / CSV: first column is the word
        first = re.split(r"[\t,]", line, maxsplit=1)[0].strip()

        # Space-separated "word freq [tag]" -> drop the trailing columns.
        # For "word pinyin freq", configure a TAB/CSV export instead.
        # Warning: without TAB/comma, a product name such as "iPhone 15"
        # would lose its trailing number; prefer TAB-separated exports.
        word = re.sub(r"\s+\d+(?:\s+[a-z]+)?$", "", first).strip()

        if word and word not in seen:
            seen.add(word)
            words.append(word)

    Path(dst).write_text("\n".join(words) + "\n", encoding="utf-8")


sogou_to_userdict("sogou_export.txt", "domain_words.txt")
```

Then:

```python
grocery = Grocery("sample", tokenizer="precise", user_dict="domain_words.txt")
print(len(grocery.feature_config.user_words))

grocery.train(train_src)
print("w:烯晶算力网络" in grocery.model.text_converter.feature2idx)
```

The words are stored in `metadata.json` under `features.user_words`, so
`Grocery("sample").load()` can reproduce them on another machine **without the
original dictionary file**.

Notes:

- Do not import a whole Sogou dictionary. Keep only domain entities, product
  names, IDs and abbreviations — a few thousand to ~20k words. Large lists bloat
  `metadata.json` and slow down `load()`.
- If `lowercase=True` (default), English words are lowercased in metadata.
  Set `lowercase=False` when product-name casing is meaningful.
- jieba's dictionary is process-global: multiple `Grocery` instances with
  different `user_dict` values share words in one process. Use separate
  processes when strict isolation is required.
- Pin the jieba version if segmentation must be byte-for-byte reproducible.
- Sogou dictionaries have their own license terms. The word list is embedded in
  the model metadata and will be redistributed with the model.
- `user_dict` only affects jieba tokenization; it is ignored when
  `custom_tokenize` is used.

### Stopwords, normalization and masking

- `stopwords` accepts a file path or iterable; it applies to word features and
  `word_boundary` character features (not to `global` character n-grams).
- `normalize="nfkc"` converts full-width letters/numbers/punctuation to half-width.
- `mask_patterns=True` replaces URLs, emails and numbers with `__url__`,
  `__email__` and `__num__`, which helps on link/price-heavy UGC.
- `char_ngram_mode="word_boundary"` slides n-grams inside words with boundary
  padding (like scikit-learn's `char_wb`), avoiding cross-word noise in
  English/mixed text. It is the only option that can noticeably slow tokenization;
  pair it with `tokenizer="precise"`.

### Efficiency

The default path is unchanged: full mode + global character n-grams, no regex or
NFKC work. 30k training takes about **26.7 s**, and 5k predictions about
**0.42 s**. Extra normalization, stopwords and user dictionaries only cost
anything when enabled.

## Measured results

Dataset: THUCNews 10-class news titles (30,000 train / 5,000 test, median length
19 characters). Same classifier and hyper-parameters (`C=8`, `max_iter=50`,
`min_df=2`); only features differ.

| Config | Full titles | Truncated to 10 chars | Train |
|---|---:|---:|---:|
| `legacy` (word 1–2gram, binary, no IDF) | 88.82% | 82.86% | 11.0 s |
| `word_tfidf` (word 1–2gram + TF-IDF) | 89.20% | 83.74% | 10.8 s |
| **`word_char` (word + char 2–3gram + TF-IDF)** | **90.22%** | **84.66%** | 26.7 s |

> These numbers are not directly comparable to the old README table
> (48k/16k titles, 32 labels, LibLinear). Run `benchmark.py` on your own data.

```bash
python benchmark.py --train cnews.train.txt --test cnews.test.txt
python benchmark.py --train cnews.train.txt --test cnews.test.txt --truncate 10
python benchmark.py --train cnews.train.txt --test cnews.test.txt \
    --train-size 30000 --test-size 5000
```

Input format is `text<TAB>label` by default; use `--label-column first` for
`label<TAB>text`.

## Options

All options are keyword-only and optional.

```python
Grocery(
    "name",
    custom_tokenize=None,  # takes precedence over tokenizer
    C=8.0,  # inverse L2 regularization strength
    class_weight=None,  # None | "balanced" | {"label": weight}
    word_ngrams=(1, 2),
    char_ngrams=(2, 3),  # () disables character n-grams
    lowercase=True,
    min_df=None,  # None = auto (1 below 1000 docs, else 2)
    max_features=None,
    binary=False,  # True reproduces the old binary features
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

Tips:

- Large corpora / limited memory: `max_features=300000` or a higher `min_df`.
- Imbalanced classes: `class_weight="balanced"`.
- Maximum speed: `char_ngrams=()`.
- English or mixed text: `tokenizer="precise", char_ngram_mode="word_boundary"`.
- Full-width or noisy UGC text: `normalize="nfkc"`, optionally `mask_patterns=True`.
- Exact legacy features: `word_ngrams=(1, 2), char_ngrams=(), binary=True,
  sublinear_tf=False, use_idf=False, normalize="none"`.

## Model format

```text
name/
    id                 # model hash
    metadata.json      # labels, feature config (tokenizer/stopwords/user words), params
    vocab.json         # feature vocabulary (UTF-8)
    vectorizer.npz     # IDF vector
    model.npz          # weight matrix + intercept
```

No `pickle`, so models can move across Python versions and operating systems.
**Old 0.1.x models are not loadable**; retrain once with 0.2.x. Early 0.2.0
models still load: missing new options fall back to the old behavior.

## Publishing to PyPI

The repository publishes automatically with GitHub Actions and PyPI Trusted
Publishing: publishing a GitHub Release triggers
`.github/workflows/pypi_publish.yml`, which builds the wheel/sdist and uploads
it to PyPI. No PyPI API token is stored in the repository or locally.

### One-time setup

1. The distribution name is `TextGrocery_Shop` (pip name `textgrocery-shop`).
   On PyPI, open **Publishing** and add a GitHub publisher with owner
   `ranvane`, repository `TextGrocery_Shop`, workflow `pypi_publish.yml`, and
   environment `pypi`. For a brand-new project, use a **pending publisher**.
2. In the GitHub repository, open **Settings → Environments** and create
   `pypi` (required reviewers are recommended).
3. Make sure `[project].version` in `pyproject.toml` matches `__version__` in
   `tgrocery/__init__.py`.

### Release flow

Prerequisite: GitHub CLI is installed and authenticated (`gh auth login`), and
CI for the commit being released has passed.

```bash
# 1. Push the code
git push origin main

# 2. Create and push the tag
git tag v0.2.0
git push origin v0.2.0

# 3. Create and publish the GitHub Release, triggering PyPI publishing
gh release create v0.2.0 \
  --title "v0.2.0" \
  --generate-notes \
  --verify-tag

# 4. List the publish workflow run
gh run list --workflow pypi_publish.yml --limit 1

# 5. Wait for the run to finish; replace RUN_ID with the ID printed above
gh run watch RUN_ID --exit-status
```

The `Upload Python Package` workflow then:

- runs `python -m build` to create
  `textgrocery_shop-0.2.0-py3-none-any.whl` and
  `textgrocery_shop-0.2.0.tar.gz`;
- uploads them to PyPI with `pypa/gh-action-pypi-publish@release/v1`.

Keep `permissions: id-token: write` in the workflow; without it OIDC cannot
exchange a short-lived PyPI credential.

### Verify the release

```bash
pip install --no-cache-dir textgrocery-shop==0.2.0
python -c "import tgrocery; print(tgrocery.__version__)"
```

PyPI versions and filenames cannot be reused. Bump the version and create a new
tag and Release before publishing again.

### Release checklist

- [ ] `pyproject.toml` and `tgrocery/__init__.py` versions match
- [ ] `ruff check .`, `ruff format --check .`, and `pytest` pass
- [ ] Code is committed and pushed, and CI passes
- [ ] A tag such as `v0.2.0` is pushed
- [ ] A GitHub Release is published
- [ ] The `Upload Python Package` workflow succeeds
- [ ] `pip install textgrocery-shop==0.2.0` works from PyPI
- [ ] `dist/`, `.pypirc`, tokens and local models are not committed

## Development

```bash
pip install -e .[test]
pytest
ruff check .
ruff format --check .
```

## Changelog

- **2026-10-24**: Renamed the PyPI distribution to `TextGrocery_Shop`
  (`pip install textgrocery-shop`); the import name stays `tgrocery`.
  Unified the license metadata and `LICENSE.txt` to BSD-3-Clause.
- **2026-10-24**: Releases now use GitHub Release + Trusted Publishing; the
  `.github/workflows/pypi_publish.yml` workflow publishes to PyPI.

## License

BSD. See [LICENSE.txt](LICENSE.txt).
