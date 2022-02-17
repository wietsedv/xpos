import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from colorama import Fore
from datasets import Dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling

if TYPE_CHECKING:
    from lib.benchmark import BenchmarkInstance
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    from transformers.data.data_collator import DataCollator


def print_instance(instance: 'BenchmarkInstance', cmd: Optional[str] = None, digest: Optional[str] = None):
    if cmd:
        print(f"{cmd} {instance.benchmark.benchmark_name}", end=" ")
    print(f"-tt={Fore.RED}{instance.task.task_type}{Fore.RESET}", end=" ")
    print(f"-tn={Fore.GREEN}{instance.task.task_name}{Fore.RESET}", end=" ")
    lang_type = "ls" if instance._source_instance is None else "lt"
    print(f"-{lang_type}={Fore.BLUE}{instance.language}{Fore.RESET}", end=" ")
    print(f"-mi={Fore.CYAN}{instance.model_id}{Fore.RESET}", end=" ")
    print(f"-mt={Fore.MAGENTA}{instance.model_config.config_name}{Fore.RESET}", end=" " if digest else "\n")
    if digest:
        print(f"-d={digest}")


def output_dir_populated(output_dir, allow_symlink: bool = False):
    output_dir = Path(output_dir)
    if not output_dir.exists() or (not allow_symlink and output_dir.is_symlink()):
        return False
    # return (output_dir / "eval_results.json").exists() or ((output_dir / "predictions").exists() and os.listdir(output_dir / "predictions"))
    return (output_dir / "pytorch_model.bin").exists()

def output_dir_ignored(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return False
    return (output_dir / ".ignore").exists()


def get_data_collator(task_type: str, tokenizer: 'PreTrainedTokenizerFast') -> 'DataCollator':
    from transformers.data.data_collator import DataCollatorForTokenClassification, DataCollatorWithPadding
    DATA_COLLATORS = {
        "masked-language-modeling": DataCollatorForLanguageModeling,
        # "text-classification": DataCollatorWithPadding,
        "token-classification": DataCollatorForTokenClassification,
        # "sentence-retrieval": DataCollatorWithPadding,
        # "question-answering": DataCollatorWithPadding,
        # "sentence-similarity": DataCollatorWithPadding,
    }
    return DATA_COLLATORS.get(task_type, DataCollatorWithPadding)(tokenizer)


def get_feature_columns(train_dataset: Dataset,
                        task_type: str) -> Tuple[Tuple[str, ...], Optional[str], Optional[List[str]]]:
    col_names = train_dataset.column_names
    label_col_name = "labels" if "labels" in col_names else "label"

    if task_type == "text-classification":
        if "premise" in col_names:
            text_col_names = "premise", "hypothesis"
        elif "sentence1" in col_names:
            text_col_names = "sentence1", "sentence2"
        else:
            text_col_names = "text",
        label_list: List[str] = train_dataset.features[label_col_name].names
        print(f"Labels: {Fore.BLUE}{label_list}{Fore.RESET}")
        return text_col_names, label_col_name, label_list

    if task_type == "token-classification":
        text_col_names = "tokens",
        label_list: List[str] = train_dataset.features[label_col_name].feature.names
        print(f"Labels: {Fore.BLUE}{label_list}{Fore.RESET}")
        return text_col_names, label_col_name, label_list

    if task_type == "question-answering":
        text_col_names = "question", "context"
        label_col_name = "answers"
        return text_col_names, label_col_name, None

    # if task_type == "sentence-similarity":
    #     text_col_names = "sentence1", "sentence2"
    #     label_col_name = "similarity_score"
    #     label_list = ["similarity"]
    #     return text_col_names, label_col_name, label_list

    if task_type == "sentence-retrieval":
        text_col_names = "sentence1", "sentence2"
        label_col_name = "similarity_score" if "similarity_score" in col_names else None
        return text_col_names, label_col_name, None

    print(train_dataset.features)
    raise ValueError(f"Unsupported task type: {task_type}")


def get_prepare_fn(tokenizer: 'PreTrainedTokenizerFast', task_type: str, label_col: Optional[str],
                   text_cols: Union[str, Tuple[str, ...]]):
    if isinstance(text_cols, str):
        text_cols = (text_cols, )

    def _tokenize(examples, **kwargs) -> Dict[str, Any]:
        return tokenizer(*map(lambda col: examples[col], text_cols),
                         max_length=tokenizer.model_max_length,
                         padding=False,
                         truncation=True,
                         is_split_into_words=task_type == "token-classification",
                         return_special_tokens_mask=False,
                         **kwargs)  # type: ignore

    if task_type == "masked-language-modeling":

        def raw_tokenize(examples, **kwargs) -> Dict[str, Any]:
            return tokenizer(*map(lambda col: examples[col], text_cols),
                             max_length=None,
                             padding=False,
                             truncation=False,
                             is_split_into_words=False,
                             return_special_tokens_mask=True,
                             **kwargs)  # type: ignore

        return raw_tokenize

    if task_type in ["text-classification", "sentence-retrieval"]:

        def tokenize_text(examples):
            tokenized_inputs = _tokenize(examples)
            if label_col is not None:
                tokenized_inputs["labels"] = examples[label_col]
            return tokenized_inputs

        return tokenize_text

    if task_type == "token-classification":

        def tokenize_tokens(examples):
            tokenized_inputs = _tokenize(examples)
            if label_col is not None:
                labels = []
                for i, label in enumerate(examples[label_col]):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)  # type: ignore
                    label_ids = []
                    prev_word_idx = None
                    for word_idx in word_ids:
                        if word_idx is None or word_idx == prev_word_idx:
                            label_ids.append(-100)
                        else:
                            label_ids.append(label[word_idx])
                        prev_word_idx = word_idx
                    labels.append(label_ids)
                tokenized_inputs["labels"] = labels
            return tokenized_inputs

        return tokenize_tokens

    if task_type == "question-answering":

        def tokenize_qa(examples):
            tokenized_inputs = _tokenize(
                examples,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")

            if task_type == "question-answering":
                tokenized_inputs["example_id"] = []
                for i in range(len(tokenized_inputs["input_ids"])):
                    sequence_ids = tokenized_inputs.sequence_ids(i)  # type: ignore

                    # One example can give several spans, this is the index of the example containing this span of text.
                    sample_index = sample_mapping[i]
                    tokenized_inputs["example_id"].append(examples["id"][sample_index])

                    # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                    # position is part of the context or not.
                    tokenized_inputs["offset_mapping"][i] = [
                        (o if sequence_ids[k] == 1 else None)
                        for k, o in enumerate(tokenized_inputs["offset_mapping"][i])
                    ]

            if label_col is not None:
                if task_type == "question-answering":
                    offset_mapping = tokenized_inputs["offset_mapping"]
                else:
                    offset_mapping = tokenized_inputs.pop("offset_mapping")

                # Let's label those examples!
                tokenized_inputs["start_positions"] = []
                tokenized_inputs["end_positions"] = []

                for i, offsets in enumerate(offset_mapping):
                    input_ids = tokenized_inputs["input_ids"][i]
                    cls_index = input_ids.index(tokenizer.cls_token_id)
                    sequence_ids = tokenized_inputs.sequence_ids(i)  # type: ignore

                    # One example can give several spans, this is the index of the example containing this span of text.
                    answers = examples[label_col][sample_mapping[i]]

                    # If no answers are given, set the cls_index as answer.
                    if len(answers["answer_start"]) == 0:
                        tokenized_inputs["start_positions"].append(cls_index)
                        tokenized_inputs["end_positions"].append(cls_index)
                        continue

                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start/end token index of the current span in the text.
                    token_start_index, token_end_index = 0, len(input_ids) - 1
                    while sequence_ids[token_start_index] != 1 or offsets[token_start_index] is None:
                        token_start_index += 1
                    while sequence_ids[token_end_index] != 1 or offsets[token_end_index] is None:
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
                        tokenized_inputs["start_positions"].append(cls_index)
                        tokenized_inputs["end_positions"].append(cls_index)
                        continue

                    # print("Init:", start_char, end_char, token_start_index, token_end_index)
                    # Move the token_start_index and token_end_index to the two ends of the answer.
                    while offsets[token_start_index] not in (None, (0, 0)) and \
                                offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    while offsets[token_end_index] not in (None, (0, 0)) and \
                                offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1

                    tokenized_inputs["start_positions"].append(token_start_index - 1)
                    tokenized_inputs["end_positions"].append(token_end_index + 1)

            return tokenized_inputs

        return tokenize_qa

    print("ERROR: Unsupported task for preprocessing: ", task_type)
    exit(1)