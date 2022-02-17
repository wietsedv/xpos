import fnmatch
import hashlib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset

from typing import TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from lib.utils import output_dir_ignored, output_dir_populated, print_instance

TASK_TYPES = [
    "masked-language-modeling",
    "text-classification",
    "token-classification",
    "question-answering",
    "sentence-similarity",
    "sentence-retrieval",
]


class BenchmarkError(Exception):
    pass


class BenchmarkKeyError(BenchmarkError):
    pass


class BenchmarkValueError(BenchmarkError):
    pass


class UnavailableInstanceError(BenchmarkError):
    pass


class NoDigestError(BenchmarkError):
    pass


def _ensure_empty_config(cls: Type, cfg: dict):
    if len(cfg) > 0:
        raise BenchmarkKeyError(f"{cls.__name__} config contains unused items:", cfg)


class BenchmarkBase:
    @classmethod
    def from_dict(cls, cfg: dict):
        _ensure_empty_config(cls, cfg)
        return cls()


@dataclass
class ModelTransform(BenchmarkBase):
    name: str
    import_instance: Optional['BenchmarkImport']
    args: Dict[str, Any]

    def get_digest(self, instance: 'BenchmarkInstance'):
        args = {**self.args}
        if self.import_instance:
            args["import_model"] = self.import_instance.load(instance).digest

        encoded = json.dumps(args, sort_keys=True).encode()
        digest = hashlib.md5(encoded).hexdigest()[:8]
        return digest

    def apply(self, instance: 'BenchmarkInstance', model: 'PreTrainedModel'):
        import lib.transform
        fn = getattr(lib.transform, self.name)
        models = [model]
        if self.import_instance:
            from transformers.models.auto import AutoModel
            extra_instance = self.import_instance.load(instance)
            extra_model = AutoModel.from_pretrained(extra_instance.output_path)
            models.append(extra_model)
        return fn(*models, **self.args)

    @classmethod
    def from_dict(cls, cfg: dict, import_dict: 'BenchmarkImportCollection'):
        name = cfg.pop("name")
        import_instance = cfg.pop("import", None)
        if import_instance:
            import_instance = import_dict[import_instance]

        args = cfg.pop("args", {})
        _ensure_empty_config(cls, cfg)
        return cls(name, import_instance, args)


@dataclass
class BenchmarkModelConfig(BenchmarkBase):
    config_name: str
    tokenizer: Optional[Union['BenchmarkImport', Dict[str, Any]]]
    model_transforms: List[ModelTransform]

    @classmethod
    def _parse_transforms(cls, transform_cfg, import_dict: 'BenchmarkImportCollection') -> List[ModelTransform]:
        if isinstance(transform_cfg, list):
            return [cls._parse_transforms(cfg, import_dict)[0] for cfg in transform_cfg]
        if isinstance(transform_cfg, dict):
            return [ModelTransform.from_dict(transform_cfg, import_dict)]
        if isinstance(transform_cfg, str):
            return [ModelTransform(transform_cfg, import_instance=None, args={})]
        raise NotImplementedError

    @classmethod
    def from_dict(cls, cfg: dict, import_dict: 'BenchmarkImportCollection'):
        tokenizer = cfg.pop("tokenizer", None)
        if isinstance(tokenizer, str):
            tokenizer = import_dict[tokenizer]

        transforms = cfg.pop("transform", [])
        transforms = cls._parse_transforms(transforms, import_dict)

        config_name = cfg.pop("_name")

        _ensure_empty_config(cls, cfg)
        return cls(config_name or "", tokenizer, transforms)


@dataclass
class BenchmarkModelType(BenchmarkBase):
    type_name: str
    train_config: BenchmarkModelConfig
    predict_config: BenchmarkModelConfig

    @classmethod
    def from_dict(cls, cfg: dict, type_dict: Dict[str, 'BenchmarkModelType'], import_dict: 'BenchmarkImportCollection'):
        type_name = cfg.pop("_name")

        # Train
        if "train" not in cfg:
            cfg["train"] = {}
        if isinstance(cfg["train"], str):
            train_config = type_dict[cfg.pop("train")].train_config
        else:
            cfg["train"]["_name"] = type_name
            train_config = BenchmarkModelConfig.from_dict(cfg.pop("train"), import_dict)

        # Predict
        if "predict" not in cfg:
            cfg["predict"] = {}
        cfg["predict"]["_name"] = type_name
        predict_config = BenchmarkModelConfig.from_dict(cfg.pop("predict"), import_dict)

        _ensure_empty_config(cls, cfg)
        return cls(type_name, train_config, predict_config)


class BenchmarModelTypeCollection(BenchmarkBase):
    type_dict: Dict[str, BenchmarkModelType]

    def __init__(self, type_dict):
        self.type_dict = type_dict

    def __getitem__(self, name: str):
        return self.type_dict[name]

    @classmethod
    def from_dict(cls, cfg: dict, import_dict: 'BenchmarkImportCollection'):
        type_dict = {}
        for type_name in list(cfg):
            type_cfg = cfg.pop(type_name)
            type_cfg["_name"] = type_name
            type_dict[type_name] = BenchmarkModelType.from_dict(type_cfg, type_dict, import_dict)

        _ensure_empty_config(cls, cfg)
        return cls(type_dict)


@dataclass
class BenchmarkImport(BenchmarkBase):
    # name: str
    config_path: Optional[str]
    task_type: Optional[str]
    task_name: Optional[str]
    language: Optional[str]
    model_id: Optional[str]
    model_type: Optional[str]
    benchmark_dict: Dict[str, 'XlingBenchmark']

    @classmethod
    def from_dict(cls, cfg: dict, benchmark_dict: Dict[str, 'XlingBenchmark']):
        # name = cfg.pop("_name")
        config_path = cfg.pop("config", None)
        language = cfg.pop("language", None)
        task_type = cfg.pop("task_type", None)
        task_name = cfg.pop("task_name", None)
        model_id = cfg.pop("model_id", None)
        model_type = cfg.pop("model_type", None)
        _ensure_empty_config(cls, cfg)
        return cls(config_path, task_type, task_name, language, model_id, model_type, benchmark_dict)

    def load(self, instance: 'BenchmarkInstance', verbose: bool = False):
        if self.config_path is None:
            benchmark = instance.benchmark
        else:
            if self.config_path not in self.benchmark_dict:
                self.benchmark_dict[self.config_path] = XlingBenchmark.from_yaml(self.config_path, verbose=verbose, strict=False)
            benchmark = self.benchmark_dict[self.config_path]

        cfg = dict(task_type=self.task_type,
                   task_name=self.task_name,
                   language_source=self.language or instance.language,
                   language_target=self.language or instance.language,
                   model_id=self.model_id or instance.model_id,
                   model_type=self.model_type)
        instances = benchmark.instances(source=True, verbose=verbose, **cfg)
        if len(instances) > 1:
            print("ERROR: Found more than one instance")
            for instance in instances:
                print_instance(instance)
            exit(1)
        if len(instances) == 0:
            raise UnavailableInstanceError(f"no instance for {self.config_path}[{cfg}]")
        return instances[0]


class BenchmarkImportCollection(BenchmarkBase):
    import_dict: Dict[str, BenchmarkImport]
    benchmark_dict: Dict[str, 'XlingBenchmark']

    def __init__(self, import_dict, benchmark_dict):
        self.import_dict = import_dict
        self.benchmark_dict = benchmark_dict

    def __getitem__(self, name: Union[str, dict]):
        if isinstance(name, dict):
            return BenchmarkImport.from_dict(name, self.benchmark_dict)

        if name not in self.import_dict:
            raise BenchmarkValueError(
                f"Import key '{name}' is not available. Choose from {list(self.import_dict.keys())}.")
        return self.import_dict[name]

    @classmethod
    def from_dict(cls, cfg: dict):
        benchmark_dict = {}
        import_dict = {}
        for import_name in list(cfg):
            import_cfg = cfg.pop(import_name)
            # import_cfg["_name"] = import_name
            import_dict[import_name] = BenchmarkImport.from_dict(import_cfg, benchmark_dict)

        _ensure_empty_config(cls, cfg)
        return cls(import_dict=import_dict, benchmark_dict=benchmark_dict)


@dataclass
class BenchmarkModel(BenchmarkBase):
    model_dict: Dict[str, List[BenchmarkModelType]]

    def models(self):
        return [(model_id, model_type) for model_id in self.model_dict for model_type in self.model_dict[model_id]]

    @classmethod
    def from_dict(cls, cfg: dict):
        import_dict = BenchmarkImportCollection.from_dict(cfg.pop("import", {}))
        type_dict = BenchmarModelTypeCollection.from_dict(cfg.pop("type"), import_dict)

        model_dict = {}
        for model_id, type_names in cfg.pop("config").items():
            model_dict[model_id] = []
            for type_name in type_names:
                model_dict[model_id].append(type_dict[type_name])

        _ensure_empty_config(cls, cfg)
        return cls(model_dict)


@dataclass
class BenchmarkTask(BenchmarkBase):
    task_type: str
    task_name: str
    task_format: str
    output_format: str
    language: List[str]
    dataset: str
    train_split: str
    validation_split: str
    test_split: str
    metric: str
    minimize: bool
    source: Optional['BenchmarkTask']

    def data_name_subset(self, language):
        args = self.dataset.format(task_name=self.task_name, language=language).split("|", maxsplit=1)
        name = args[0]
        subset = args[1] if len(args) > 1 else None
        return name, subset

    def load_dataset(self, language: str, split: str):
        name, subset = self.data_name_subset(language)

        if split == "train":
            split = self.train_split
        elif split == "validation":
            split = self.validation_split
        elif split == "test":
            split = self.test_split
        else:
            raise ValueError(split)

        dataset = load_dataset(name, subset, split=split)
        assert isinstance(dataset, Dataset)
        return dataset

    @classmethod
    def from_dict(cls, cfg: dict, source_tasks: Optional['BenchmarkTaskCollection'] = None):
        task_name = cfg.pop("name")

        source_name = cfg.pop("source", task_name)
        source = None
        if source_tasks and source_name is not None:
            source = source_tasks.get(source_name, None)
            if source is None:
                if source_name != task_name:
                    raise BenchmarkValueError(f"Unknown source task: {source_name}")
                source_cfg = cfg.copy()
                source_cfg["name"] = task_name
                if source_tasks.language:
                    source_cfg["language"] = [l for l in source_tasks.language if l in source_cfg["language"]]
                source = BenchmarkTask.from_dict(source_cfg)

        task_format = cfg.pop("task_format")
        output_format = cfg.pop("output_format")
        task_type = cfg.pop("type")
        language = cfg.pop("language")
        dataset = cfg.pop("dataset", "{task_name}")
        train_split = cfg.pop("train_split", "train")
        validation_split = cfg.pop("validation_split", "validation")
        test_split = cfg.pop("test_split", "test")
        metric = cfg.pop("metric", "accuracy")
        minimize = cfg.pop("minimize", False)

        _ensure_empty_config(cls, cfg)
        return cls(
            task_type=task_type,
            task_name=task_name,
            task_format=task_format,
            output_format=output_format,
            language=language,
            dataset=dataset,
            train_split=train_split,
            validation_split=validation_split,
            test_split=test_split,
            metric=metric,
            minimize=minimize,
            source=source,
        )


class BenchmarkTaskCollection(BenchmarkBase):
    task_dict: Dict[str, BenchmarkTask]
    language: Optional[List[str]]

    def __init__(self, task_dict, language):
        self.task_dict = task_dict
        self.language = language

    def __getitem__(self, name: str):
        return self.task_dict[name]

    def get(self, name: str, default: Any):
        return self.task_dict.get(name, default)

    def tasks(self):
        return list(self.task_dict.values())

    def __repr__(self) -> str:
        return f"BenchmarkTaskCollection({self.task_dict})"

    @classmethod
    def from_dict(cls, cfg: dict, source_tasks: Optional['BenchmarkTaskCollection'] = None):
        task_format = cfg.pop("task_format", "{task_name}-{language}")
        output_format = cfg.pop("output_format", "{source}_{model_type}_{task_name}")
        dataset = cfg.pop("dataset", None)
        language = cfg.pop("language", None)
        if type(language) == str:
            language = [language]

        task_dict = {}
        for task_type in list(cfg):
            if task_type in TASK_TYPES:
                for task_name, task_cfg in cfg.pop(task_type).items():
                    if "name" not in task_cfg:
                        task_cfg["name"] = task_name
                    if "task_format" not in task_cfg:
                        task_cfg["task_format"] = task_format
                    if "output_format" not in cfg:
                        task_cfg["output_format"] = output_format
                    if "dataset" not in cfg and dataset:
                        task_cfg["dataset"] = dataset

                    if "type" not in task_cfg:
                        task_cfg["type"] = task_type
                    if language:
                        if "language" in task_cfg:
                            task_cfg["language"] = [l for l in language if l in task_cfg["language"]]
                        else:
                            task_cfg["language"] = language
                    task_dict[task_name] = BenchmarkTask.from_dict(task_cfg, source_tasks)

        _ensure_empty_config(cls, cfg)
        return cls(task_dict, language)


@dataclass
class BenchmarkSource(BenchmarkBase):
    task: BenchmarkTaskCollection

    @classmethod
    def from_dict(cls, cfg: dict):
        task = BenchmarkTaskCollection.from_dict(cfg)

        _ensure_empty_config(cls, cfg)
        return cls(task)


@dataclass
class BenchmarkTarget(BenchmarkBase):
    task: BenchmarkTaskCollection

    def tasks(self):
        return self.task.tasks()

    def languages(self):
        return self.task.language

    @classmethod
    def from_dict(cls, cfg: dict, source: BenchmarkSource):
        task = BenchmarkTaskCollection.from_dict(cfg, source.task)

        _ensure_empty_config(cls, cfg)
        return cls(task)


@dataclass
class BenchmarkInstance:
    benchmark: 'XlingBenchmark'
    task: BenchmarkTask
    language: str
    model_id: str
    model_config: BenchmarkModelConfig
    training_digest: Optional[str] = None
    _source_instance: Optional['BenchmarkInstance'] = None

    @property
    def source_instance(self) -> 'BenchmarkInstance':
        return self._source_instance or self

    @property
    def digest(self):
        if not self._source_instance:
            return self.training_digest or self.set_best_digest()[0]

        digest = self._source_instance.digest

        # Transformations
        transform_digests = [transform.get_digest(self) for transform in self.model_config.model_transforms]
        if transform_digests:
            transform_digest = hashlib.md5("".join(transform_digests).encode()).hexdigest()[:8]
            digest = f"{digest}_{transform_digest}"

        return digest

    def load_datasets(self, split=None) -> Union[DatasetDict, Dataset]:
        source = self._source_instance is None

        if split is not None:
            return self.task.load_dataset(self.language, split)

        if source:
            train_dataset = self.task.load_dataset(self.language, "train")
            try:
                validation_dataset = self.task.load_dataset(self.language, "validation")
            except ValueError:
                return DatasetDict(train=train_dataset)
            return DatasetDict(train=train_dataset, validation=validation_dataset)

        test_dataset = self.task.load_dataset(self.language, "test")
        return DatasetDict(test=test_dataset)

    def load_tokenizer(self, dataset: Dataset = None) -> 'PreTrainedTokenizerFast':
        from transformers import AutoTokenizer

        tokenizer_kwargs, vocab_size = {}, None
        if self.model_config.tokenizer:
            if isinstance(self.model_config.tokenizer, BenchmarkImport):
                # tokenizer_instance = _import_instance(self.model_config.tokenizer, self)
                tokenizer_path = self.model_config.tokenizer.load(self).output_path
                return AutoTokenizer.from_pretrained(tokenizer_path)

            tokenizer_kwargs = {k: v for k, v in self.model_config.tokenizer.items()}
            vocab_size = tokenizer_kwargs.pop("vocab_size", None)

        custom_tokenizer_path = None
        if vocab_size is not None:
            fingerprint = dataset._fingerprint if dataset else "UNKNOWN"
            custom_tokenizer_path = self.output_base / f"tokenizer-{fingerprint}-{vocab_size:_}"
            if not custom_tokenizer_path.exists():
                custom_tokenizer_paths = list(self.output_base.glob(f"tokenizer-*-{vocab_size:_}"))
                if custom_tokenizer_paths:
                    custom_tokenizer_path = custom_tokenizer_paths[0]
                    print(
                        f"WARNING: fingerprint of tokenizer does not match: {custom_tokenizer_path} != {fingerprint}"
                    )
            if custom_tokenizer_path.exists():
                return AutoTokenizer.from_pretrained(custom_tokenizer_path)

        model_id = self.output_path if self._source_instance else self.model_id
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)

        if custom_tokenizer_path:

            def batch_iterator(batch_size=1000):
                for i in range(0, len(dataset), batch_size):
                    yield dataset[i:i + batch_size]["text"]  # type: ignore

            tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size)
            tokenizer.save_pretrained(custom_tokenizer_path)

        return tokenizer

    @property
    def model_path_(self):
        return self._source_instance.output_path if self._source_instance else self.model_id
    
    @property
    def model_path(self):
        return self.model_path_ if self.model_config.tokenizer is None and len(self.model_config.model_transforms) == 0 else None

    def load_model_(self, labels: Optional[List[str]] = None, verbose: bool = False, **kwargs):
        """ Load model without transformations """
        from transformers import (AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification,
                                  AutoModelForTokenClassification, AutoModelForMaskedLM)
        MODEL_TYPES = {
            "masked-language-modeling": AutoModelForMaskedLM,
            "text-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "question-answering": AutoModelForQuestionAnswering,
            "sentence-similarity": AutoModelForSequenceClassification,
            "sentence-retrieval": AutoModel,
        }
        model_id = self.model_path_
        not verbose or print("loading model:", model_id)
        if labels is not None:
            kwargs["num_labels"] = len(labels)
        model: PreTrainedModel = MODEL_TYPES[self.task.task_type].from_pretrained(model_id, **kwargs)
        return model

    def load_model(self, labels: Optional[List[str]] = None, verbose: bool = False, no_transform: bool = False, **kwargs) -> 'PreTrainedModel':
        """ Load model with optional transformations """
        model = self.load_model_(labels, verbose=verbose, **kwargs)

        # Resize word embeddings
        if self._source_instance is None and self.model_config.tokenizer:
            if not isinstance(self.model_config.tokenizer, BenchmarkImport):
                assert isinstance(self.model_config.tokenizer, dict)
                vocab_size = self.model_config.tokenizer.get("vocab_size", None)
                if vocab_size is not None:
                    model.resize_token_embeddings(vocab_size)
                    # model._init_weights(model.get_input_embeddings())
                    # model.tie_weights()
                    not verbose or print("lexical layer is resized to", vocab_size)

        # Apply model transformations
        if self.model_config.model_transforms and not no_transform:
            for transform in self.model_config.model_transforms:
                transform.apply(self, model)

        # Set label maps
        model.config.architectures = [model.__class__.__name__]
        if labels:
            model.config.id2label = dict(enumerate(labels))
            model.config.label2id = {l: i for i, l in model.config.id2label.items()}
        return model

    def output_dirs(self, only_populated: bool = True, include_symlinks: bool = False, include_ignored: bool = False):
        if not self.output_base.exists():
            return []
        output_dirs = self.output_base.glob("*")
        if not only_populated:
            return list(output_dirs)
        return [
            output_dir for output_dir in output_dirs
            if output_dir_populated(output_dir, allow_symlink=include_symlinks) or (
                include_ignored and output_dir_ignored(output_dir))
        ]

    def get_training_digests(self, only_populated=True, include_symlinks: bool = False, include_ignored: bool = False):
        training_digests = list(
            map(
                lambda x: x.name,
                self.output_dirs(only_populated=only_populated,
                                 include_symlinks=include_symlinks,
                                 include_ignored=include_ignored)))
        return training_digests

    def get_score(self, training_digest: Optional[str] = None, train: bool = False) -> Optional[float]:
        training_digest = training_digest or self.training_digest
        if training_digest is None:
            raise ValueError("Training digest is unknown")
        
        split = "train" if train else "eval"
        results_path = self.output_base / training_digest / f"{split}_results.json"
        if not results_path.exists():
            return
        with open(results_path) as f:
            eval_results = json.load(f)

        metric = self.task.metric
        if f"{split}_{metric}" not in eval_results:
            short_metric = metric.split("_")[-1]
            if f"{split}_{short_metric}" in eval_results:
                metric = short_metric
            else:
                print(f"ERROR: metric {metric} is not available in {results_path}")
                exit(1)

        score = eval_results[f"{split}_{metric}"]
        if isinstance(score, dict):
            score = score["score"]
        return score

    def set_best_digest(self):
        if self._source_instance:
            best_training_digest, best_score = self._source_instance.set_best_digest()
            self.training_digest = best_training_digest
            return best_training_digest, best_score

        minimize = self.task.minimize

        best_training_digest, best_score = None, np.inf if minimize else -np.inf
        for training_digest in self.get_training_digests():
            score = self.get_score(training_digest)
            if score is None:
                continue

            if score != best_score and minimize == (score < best_score):
                best_training_digest, best_score = training_digest, score

        self.training_digest = best_training_digest
        return best_training_digest, best_score if best_training_digest is not None else np.nan

    def _format(self, name: str, source: Optional[str] = None, model_type: Optional[str] = None):
        format_kwargs = {
            "source_language":
            self.source_instance.language,
            "source":
            source or self.model_id.replace("/", "-"),
            "task_name":
            self.task.task_format.format(task_name=self.task.task_name.replace("_", "-"), language=self.language),
            "language":
            self.language,
            "model_type":
            model_type or self.model_config.config_name,
        }
        # format_kwargs["model_type"] = self.model_config.config_name

        name = name.format(**format_kwargs)
        name = re.sub(r"(-*_-*)+", "_", name)
        name = re.sub(r"-+", "-", name)
        name = re.sub(r"^[-_]+|[-_]+$", "", name)
        return name

    @property
    def output_name(self) -> str:
        if self._source_instance:
            _fmt_fn = self._source_instance._format
        else:
            _fmt_fn = self._format

        return _fmt_fn(self.task.output_format, model_type=self.model_config.config_name)

    @property
    def output_base(self):
        return Path(self.benchmark.output_base_fmt.format(output_name=self.output_name))

    @property
    def output_path(self) -> Path:
        digest = self.digest
        if not digest:
            raise NoDigestError(self.output_base)
        return self.output_base / digest

    @property
    def predict_name(self):
        if not self._source_instance:
            raise BenchmarkError("not a prediction instance")
        return self._format(self.task.output_format, source="")

    def get_predict_path(self, split: str = "test", ext: Optional[str] = "tsv"):
        if not self._source_instance:
            raise BenchmarkError("not a prediction instance")
        predict_filename = self._format(self.task.output_format, source=split)
        if ext:
            predict_filename = f"{predict_filename}.{ext}"
        return Path(
            self.benchmark.predict_path_fmt.format(output_path=self.output_path, predict_filename=predict_filename))


@dataclass
class XlingBenchmark(BenchmarkBase):
    benchmark_name: str
    output_base_fmt: str
    predict_path_fmt: str
    target: BenchmarkTarget
    model: BenchmarkModel

    @staticmethod
    def _filter(cfg_key: Optional[str], value: Union[str, list]):
        """ WARNING: Edits lists in-place """
        if cfg_key is None:
            return True

        # list
        if isinstance(value, list):
            if "," in cfg_key:
                new_value = [x for x in value if x in cfg_key.split(",")]
                value.clear()
                value.extend(new_value)
                return len(new_value) > 0
            if cfg_key not in value:
                return False
            value.clear()
            value.append(cfg_key)
            return True

        # string
        if "," in cfg_key:
            for cfg_key_ in cfg_key.split(","):
                if XlingBenchmark._filter(cfg_key_, value):
                    return True
            return False
        return fnmatch.fnmatch(value, cfg_key)

    def instances_(self, source: bool, cfg: dict):
        instances: List[BenchmarkInstance] = []
        output_ids = set()

        for predict_task in self.target.tasks():
            assert predict_task.source is not None

            languages_source = [*predict_task.source.language]
            if len(languages_source) == 0:
                raise BenchmarkValueError(f"No source languages for task {predict_task.task_name}")

            if not (\
                self._filter(cfg.get("task_type"), predict_task.task_type) and \
                self._filter(cfg.get("task_name"), predict_task.source.task_name if source else predict_task.task_name) and \
                self._filter(cfg.get("language_source"), languages_source)):
                continue

            languages_target = [*predict_task.language]
            if len(languages_target) == 0:
                print(f"WARNING: no target languages for task {predict_task.task_name}")
                continue

            if not source and not self._filter(cfg.get("language_target"), languages_target):
                continue

            models = self.model.models()
            if len(models) == 0:
                raise BenchmarkValueError(f"no models for task {predict_task.task_name}")

            for model_id, model_type in models:
                if not (\
                    self._filter(cfg.get("model_id"), model_id) and \
                    self._filter(cfg.get("model_type"), model_type.type_name)):
                    continue

                for language_source in languages_source:
                    source_instance = BenchmarkInstance(
                        benchmark=self,
                        task=predict_task.source,
                        language=language_source,
                        model_id=model_id,
                        model_config=model_type.train_config,
                    )

                    if source:
                        output_id = source_instance.output_name
                        if output_id not in output_ids:
                            instances.append(source_instance)
                            output_ids.add(output_id)
                        continue

                    for language_target in languages_target:
                        target_instance = BenchmarkInstance(
                            benchmark=self,
                            task=predict_task,
                            language=language_target,
                            model_id=model_id,
                            model_config=model_type.predict_config,
                            _source_instance=source_instance,
                        )
                        output_id = target_instance.output_name, target_instance.predict_name
                        if output_id not in output_ids:
                            instances.append(target_instance)
                            output_ids.add(output_id)
        return instances

    def instances(self,
                  source: bool = False,
                  args: Optional[Namespace] = None,
                  **cfg) -> List[BenchmarkInstance]:
        if args:
            cfg["task_type"] = cfg.get("task_type", args.task_type)
            cfg["task_name"] = cfg.get("task_name", args.task_name)
            cfg["language_source"] = cfg.get("language_source", args.language_source)
            cfg["language_target"] = cfg.get("language_target", args.language_target)
            cfg["model_id"] = cfg.get("model_id", args.model_id)
            cfg["model_type"] = cfg.get("model_type", args.model_type)

        instances = self.instances_(source, cfg)
        return instances

    @classmethod
    def from_dict(cls, cfg: dict, strict: bool = True, verbose: bool = False):
        benchmark_name = cfg.pop("name")
        output_base_fmt = cfg.pop("output_base", "./output/{output_name}")
        predict_path_fmt = cfg.pop("predict_path", "{output_path}/predictions/{predict_filename}")

        source = BenchmarkSource.from_dict(cfg.pop("source", {}))
        not verbose or print(f"Parsed `source` {source.task.language=}")

        target = BenchmarkTarget.from_dict(cfg.pop("target", {}), source)
        not verbose or print(f"Parsed `target` {target.task.language=}")

        model = BenchmarkModel.from_dict(cfg.pop("model"))
        not verbose or print("Parsed `model`")

        _ensure_empty_config(cls, cfg)
        self = cls(
            benchmark_name=benchmark_name,
            output_base_fmt=output_base_fmt,
            predict_path_fmt=predict_path_fmt,
            target=target,
            model=model,
        )
        if strict:
            train_names = list(map(lambda x: x.output_name, self.instances(source=True)))
            if len(train_names) != len(set(train_names)):
                train_dups = {name for name in train_names if train_names.count(name) > 1}
                raise BenchmarkValueError(f"list of train ids contains duplicates: {train_dups}")

            predict_ids = list(map(lambda x: (x.output_name, x.predict_name), self.instances()))
            if len(predict_ids) != len(set(predict_ids)):
                predict_dups = {pid for pid in predict_ids if predict_ids.count(pid) > 1}
                raise BenchmarkValueError(f"list of prediction output names contains duplicates: {predict_dups}")
        not verbose or print("Parsed config")
        return self

    @staticmethod
    def from_yaml(path: str, strict: bool = True, verbose: bool = False):
        import yaml

        if not os.path.exists(path):
            path = os.path.join("configs", str(path) + ".yaml")

        with open(path) as f:
            cfg = yaml.safe_load(f)

        if "name" not in cfg:
            cfg["name"] = os.path.basename(os.path.splitext(path)[0])

        return XlingBenchmark.from_dict(cfg, strict=strict, verbose=verbose)

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser):
        group = parser.add_argument_group("benchmark")
        group.add_argument("config")
        group.add_argument("-tt", "--task_type", default=None)
        group.add_argument("-tn", "--task_name", default=None)
        group.add_argument("-ls", "--language_source", default=None)
        group.add_argument("-lt", "--language_target", default=None)
        group.add_argument("-mi", "--model_id", default=None)
        group.add_argument("-mt", "--model_type", default=None)
