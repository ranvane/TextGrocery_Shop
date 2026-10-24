"""TextGrocery_Shop 单元测试。

覆盖范围：

* 基础 API：训练 -> 预测 -> 测试；
* 模型保存 / 加载（新目录格式）；
* 文件输入（UTF-8、制表符分隔）；
* 自定义分词器；
* 未训练时的异常；
* 字符 n-gram / TF-IDF 等新配置；
* 结果对象的指标计算。
"""

from __future__ import annotations

import json
import shutil

import pytest

from tgrocery import Grocery, GroceryException, GroceryNotTrainException
from tgrocery.base import GroceryPredictResult, GroceryTestResult, read_text_src
from tgrocery.converter import FeatureConfig, GroceryTextConverter

TRAIN_SRC = [
    ("education", "名师指导托福语法技巧：名词的复数形式"),
    ("education", "中国高考成绩海外认可 是狼来了吗"),
    ("education", "考生必读：新托福写作考试评分标准"),
    ("education", "考研英语复习指南 词汇阅读是关键"),
    ("sports", "图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼"),
    ("sports", "四川丹棱举行全国长距登山挑战赛 近万人参与"),
    ("sports", "意甲首轮补赛交战记录 米兰客场8战不败国米10年连胜"),
    ("sports", "国足客场战胜对手 球迷欢呼"),
    ("tech", "苹果发布新款手机 搭载自研芯片"),
    ("tech", "华为推出新一代处理器 性能提升"),
    ("tech", "人工智能大模型 开源生态快速发展"),
    ("tech", "新款笔记本发布 续航大幅提升"),
]


@pytest.fixture()
def grocery(tmp_path, monkeypatch):
    """在临时目录中训练一个模型，避免污染仓库。"""
    monkeypatch.chdir(tmp_path)
    model = Grocery("demo")
    model.train(TRAIN_SRC)
    return model


def test_train_predict(grocery):
    """训练后应能正确预测三类文本。"""
    assert grocery.get_load_status()
    assert str(grocery.predict("托福写作考试评分标准")) == "education"
    assert str(grocery.predict("米兰客场战胜国米")) == "sports"
    assert str(grocery.predict("华为发布自研芯片")) == "tech"


def test_predict_result_fields(grocery):
    """预测结果应包含标签、决策值与概率，且概率之和为 1。"""
    result = grocery.predict("托福写作考试评分标准")
    assert result.predicted_y == "education"
    assert set(result.dec_values) == {"education", "sports", "tech"}
    assert set(result.probabilities) == {"education", "sports", "tech"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    # __str__ 与标签一致，保证 print(grocery.predict(...)) 的旧用法不变
    assert str(result) == "education"


def test_test_accuracy(grocery):
    """在训练集上评估应接近 100%。"""
    result = grocery.test(TRAIN_SRC)
    assert result.accuracy_overall == pytest.approx(1.0)
    assert set(result.accuracy_labels) == {"education", "sports", "tech"}
    assert "accuracy" in result.to_table()


def test_save_and_load(grocery, tmp_path):
    """保存后应生成新格式文件，重新加载后预测结果一致。"""
    grocery.save()
    model_dir = tmp_path / "demo"
    assert (model_dir / "metadata.json").exists()
    assert (model_dir / "vocab.json").exists()
    assert (model_dir / "vectorizer.npz").exists()
    assert (model_dir / "model.npz").exists()
    assert (model_dir / "id").exists()

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format_version"] == 2
    assert metadata["features"]["char_ngrams"] == [2, 3]
    assert set(metadata["labels"]) == {"education", "sports", "tech"}

    reloaded = Grocery("demo")
    reloaded.load()
    assert reloaded.get_load_status()
    assert str(reloaded.predict("托福写作考试评分标准")) == "education"
    assert str(reloaded.predict("意甲米兰客场")) == "sports"


def test_file_input(tmp_path, monkeypatch):
    """支持 UTF-8、制表符分隔的文件输入。"""
    monkeypatch.chdir(tmp_path)
    train_file = tmp_path / "train.tsv"
    train_file.write_text(
        "\n".join(f"{label}\t{text}" for label, text in TRAIN_SRC),
        encoding="utf-8",
    )
    test_file = tmp_path / "test.tsv"
    test_file.write_text(
        "education\t托福写作考试评分标准\nsports\t米兰客场战胜国米\n",
        encoding="utf-8",
    )

    model = Grocery("file_model")
    model.train(str(train_file))
    result = model.test(str(test_file))
    assert result.accuracy_overall == pytest.approx(1.0)

    samples = read_text_src(str(train_file))
    assert samples[0] == ["education", "名师指导托福语法技巧：名词的复数形式"]


def test_custom_tokenize(tmp_path, monkeypatch):
    """自定义分词器（逐字符切分）应可正常训练与预测。"""
    monkeypatch.chdir(tmp_path)
    model = Grocery("char_model", custom_tokenize=list)
    model.train(TRAIN_SRC)
    assert str(model.predict("托福写作考试评分标准")) == "education"


def test_feature_options(tmp_path, monkeypatch):
    """关闭字符 n-gram / 使用二值特征时仍应可训练。"""
    monkeypatch.chdir(tmp_path)
    model = Grocery(
        "legacy_like",
        word_ngrams=(1, 2),
        char_ngrams=(),
        binary=True,
        sublinear_tf=False,
        use_idf=False,
    )
    model.train(TRAIN_SRC)
    label = str(model.predict("法网孟菲尔斯苦战"))
    assert label in {"education", "sports", "tech"}


def test_untrained_raises(tmp_path, monkeypatch):
    """未训练时 predict/test 应抛出明确的异常。"""
    monkeypatch.chdir(tmp_path)
    model = Grocery("empty")
    with pytest.raises(GroceryNotTrainException):
        model.predict("任意文本")
    with pytest.raises(GroceryNotTrainException):
        model.test(TRAIN_SRC)


def test_single_class(tmp_path, monkeypatch):
    """只有一个类别时应返回常量分类器，而不是崩溃。"""
    monkeypatch.chdir(tmp_path)
    model = Grocery("single").train([("only", "任意文本"), ("only", "另一个文本")])
    assert str(model.predict("完全无关的内容")) == "only"


def test_empty_text_and_bytes(grocery):
    """空文本与 UTF-8 字节串都应可预测（旧代码可能传 bytes）。"""
    assert str(grocery.predict("")) in {"education", "sports", "tech"}
    assert str(grocery.predict("托福考试".encode())) == "education"


def test_save_twice_overwrites(grocery):
    """重复 save() 应覆盖旧目录，而不是报错。"""
    grocery.save()
    grocery.save()
    reloaded = Grocery("demo")
    reloaded.load()
    assert reloaded.get_load_status()


def test_invalid_options():
    """非法参数应在构造时立即报错。"""
    with pytest.raises(GroceryException):
        Grocery("bad", custom_tokenize=123)
    with pytest.raises(GroceryException):
        Grocery("bad", C=0)
    with pytest.raises(ValueError):
        Grocery("bad", char_ngrams=(-1,))
    with pytest.raises(ValueError):
        Grocery("bad", tokenizer="unknown")
    with pytest.raises(ValueError):
        Grocery("bad", char_ngram_mode="unknown")
    with pytest.raises(ValueError):
        Grocery("bad", normalize="unknown")


def test_nfkc_normalization():
    """全角字母/数字应与半角产生相同特征。"""
    converter = GroceryTextConverter(
        config=FeatureConfig(word_ngrams=(), char_ngrams=(2, 3), normalize="nfkc")
    )
    assert converter._extract_counts("ＡＢＣ１２３") == converter._extract_counts("abc123")


def test_mask_patterns():
    """URL / 邮箱 / 数字应折叠成稳定占位符。"""
    converter = GroceryTextConverter(
        custom_tokenize=str.split,
        config=FeatureConfig(word_ngrams=(1,), char_ngrams=(), mask_patterns=True),
    )
    counts = converter._extract_counts("价格123元 邮箱a@b.com 链接https://x.cn/1")
    assert "w:__num__" in counts
    assert "w:__email__" in counts
    assert "w:__url__" in counts


def test_tokenizer_modes_differ():
    """全模式与精确模式应产生不同的词特征。"""
    full = GroceryTextConverter(
        config=FeatureConfig(word_ngrams=(1,), char_ngrams=(), tokenizer="full")
    )
    precise = GroceryTextConverter(
        config=FeatureConfig(word_ngrams=(1,), char_ngrams=(), tokenizer="precise")
    )
    text = "研究生命起源"
    assert full._extract_counts(text) != precise._extract_counts(text)


def test_user_dict_and_stopwords():
    """用户词典能补充未登录词，停用词会从词特征中移除。"""
    converter = GroceryTextConverter(
        config=FeatureConfig(
            word_ngrams=(1,),
            char_ngrams=(),
            tokenizer="precise",
            user_words=("烯晶算力网络",),
            stopwords=("平台",),
        )
    )
    counts = converter._extract_counts("烯晶算力网络平台提供服务")
    assert "w:烯晶算力网络" in counts
    assert "w:平台" not in counts


def test_char_ngram_word_boundary():
    """word_boundary 模式不应生成跨词字符 n-gram。"""
    wb = GroceryTextConverter(
        config=FeatureConfig(
            word_ngrams=(),
            char_ngrams=(2,),
            char_ngram_mode="word_boundary",
            tokenizer="precise",
        )
    )
    global_mode = GroceryTextConverter(
        config=FeatureConfig(
            word_ngrams=(),
            char_ngrams=(2,),
            char_ngram_mode="global",
        )
    )
    wb_counts = wb._extract_counts("ab cd")
    assert "c:ab" in wb_counts
    assert "c:bc" not in wb_counts
    assert "c:bc" in global_mode._extract_counts("ab cd")


def test_new_config_persisted(tmp_path, monkeypatch):
    """新增配置应写入 metadata，并在 load 后恢复。"""
    monkeypatch.chdir(tmp_path)
    grocery = Grocery(
        "cfg",
        tokenizer="precise",
        normalize="none",
        char_ngram_mode="word_boundary",
        mask_patterns=True,
        stopwords=["的", "平台"],
        user_dict=["烯晶算力网络"],
    )
    grocery.train(TRAIN_SRC)
    grocery.save()

    meta = json.loads((tmp_path / "cfg" / "metadata.json").read_text(encoding="utf-8"))
    features = meta["features"]
    assert features["tokenizer"] == "precise"
    assert features["normalize"] == "none"
    assert features["char_ngram_mode"] == "word_boundary"
    assert features["mask_patterns"] is True
    assert features["stopwords"] == ["的", "平台"]
    assert features["user_words"] == ["烯晶算力网络"]

    loaded = Grocery("cfg")
    loaded.load()
    assert loaded.feature_config.tokenizer == "precise"
    assert loaded.feature_config.char_ngram_mode == "word_boundary"
    assert str(loaded.predict("法网孟菲尔斯苦战")) in {"education", "sports", "tech"}
    counts = loaded.model.text_converter._extract_counts("烯晶算力网络平台提供服务")
    assert "w:烯晶算力网络" in counts
    assert "w:平台" not in counts


def test_word_list_files(tmp_path, monkeypatch):
    """user_dict / stopwords 也支持 UTF-8 文件路径。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "words.txt").write_text("烯晶算力网络\n\n星云一体机\n", encoding="utf-8")
    (tmp_path / "stop.txt").write_text("平台\n", encoding="utf-8")
    grocery = Grocery(
        "files",
        user_dict=tmp_path / "words.txt",
        stopwords=tmp_path / "stop.txt",
    )
    assert grocery.feature_config.user_words == ("烯晶算力网络", "星云一体机")
    assert grocery.feature_config.stopwords == ("平台",)

    with pytest.raises(GroceryException):
        Grocery("missing", user_dict=tmp_path / "nope.txt")


def test_old_metadata_defaults():
    """缺少新字段的旧模型元数据应回退到旧行为。"""
    config = FeatureConfig.from_dict({"word_ngrams": [1, 2], "char_ngrams": [2, 3]})
    assert config.tokenizer == "full"
    assert config.normalize == "none"
    assert config.char_ngram_mode == "global"
    assert config.stopwords == ()
    assert config.user_words == ()


def test_empty_training_data(tmp_path, monkeypatch):
    """空训练集应给出明确异常。"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(GroceryException):
        Grocery("empty").train([])


def test_test_result_metrics():
    """结果对象的准确率 / 精确率 / 召回率计算。"""
    result = GroceryTestResult(
        true_y=["a", "a", "b", "b"],
        predicted_y=["a", "b", "b", "b"],
    )
    assert result.accuracy_overall == pytest.approx(0.75)
    assert result.accuracy_labels["a"] == pytest.approx(1.0)
    assert result.recall_labels["a"] == pytest.approx(0.5)
    assert result.accuracy_labels["b"] == pytest.approx(2 / 3)


def test_predict_result_accepts_mapping():
    """兼容旧式用法：dec_values 直接传 {标签: 决策值} 字典。"""
    result = GroceryPredictResult("a", {"a": 1.0, "b": 2.0})
    assert result.dec_values == {"a": 1.0, "b": 2.0}
    assert result.labels == ["a", "b"]
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_delimiter_in_text(tmp_path):
    """正文中再次出现制表符时不应丢样本（split 限制为一次）。"""
    path = tmp_path / "weird.tsv"
    path.write_text("sports\t比分\t3:1\n", encoding="utf-8")
    samples = read_text_src(str(path))
    assert samples[0][0] == "sports"
    assert samples[0][1] == "比分\t3:1"


def test_model_dir_is_removed(grocery, tmp_path):
    """确保测试结束后目录可被清理（验证没有遗留文件句柄）。"""
    grocery.save()
    shutil.rmtree(tmp_path / "demo")
    assert not (tmp_path / "demo").exists()
