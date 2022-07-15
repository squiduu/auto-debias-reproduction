import sys
import os
import random
from typing import Dict, List
import numpy as np
from datasets.load import load_dataset, load_metric
from logging import Logger
from transformers import BatchEncoding, Trainer
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import set_seed, EvalPrediction, get_last_checkpoint
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import default_data_collator, DataCollatorWithPadding
from transformers.trainer import Trainer
from glue_utils import clear_console, get_glue_logger
from glue_config import TASK_TO_KEYS, DataArguments, ModelArguments


def run_glue(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    logger: Logger,
):
    # log args
    logger.info(f"Model args: {vars(model_args)}")
    logger.info(f"Data args: {vars(data_args)}")
    logger.info(f"Training args: {vars(training_args)}")

    # set seed before initialize model
    logger.info(f"Set seed.")
    set_seed(training_args.seed)

    logger.info("Detecting last checkpoint.")
    last_ckpt = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_ckpt is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_ckpt}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if data_args.task_name is not None:
        logger.info("Download a GLUE dataset from HuggingFace hub.")
        raw_datasets = load_dataset(path="glue", name=data_args.task_name, cache_dir=model_args.cache_dir)
        # clean up all cache files except the currently used cache file
        raw_datasets.cleanup_cache_files()

    # set labels
    if data_args.task_name is not None:
        logger.info(f"Set labels.")
        is_regression = True if data_args.task_name == "stsb" else False
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

    # load a pre-trained model and tokenizer
    logger.info(f"Load a pre-trained model and tokenizer.")
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # load biased checkpoint model or pre-trained model from huggingface
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # preprocess the raw datasets
    logger.info(f"Get keys from the task.")
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = TASK_TO_KEYS[data_args.task_name]

    # set sequence padding strategy
    logger.info("Set padding strategy.")
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # some models have set the order of the labels to use, so let's make sure we do use it
    label2id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # some have all caps in their config, some do not
        labelname2id = {str.lower(k): v for k, v in dict.items(model.config.label2id)}
        if list(sorted(labelname2id.keys())) == list(sorted(label_list)):
            label2id = {i: int(labelname2id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(labelname2id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    logger.info("Set model config according to the datasets labels.")
    if label2id is not None:
        model.config.label2id = label2id
        model.config.id2label = {id: label for id, label in dict.items(config.label2id)}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {label: i for i, label in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in dict.items(config.label2id)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    logger.info(f"Set the maximum sequence length.")
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_glue(examples: Dict[str, List]) -> BatchEncoding:
        """Create single or pair sentence inputs and tokenize examples of GLUE dataset.

        Args:
            examples (Dict[str, List]): Iterable examples in GLUE dataset.

        Returns:
            result (BatchEncoding): A tokenized result with added labels.
        """
        # get inputs as tuple of examples -> (sent1,) or (sent1, sent2)
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        # tokenize the input text -> ([CLS] sent1 [SEP]) or ([CLS] sent1 [SEP] sent2 [SEP])
        result = tokenizer.__call__(*inputs, padding=padding, max_length=max_seq_length, truncation=True)

        # map labels to ids (not necessary for GLUE tasks)
        if label2id is not None and "label" in examples:
            result["label"] = [(label2id[label] if label != -1 else -1) for label in examples["label"]]

        return result

    logger.info("Preprocess the dataset.")
    with training_args.main_process_first(desc="dataset map preprocessing."):
        # apply a preprocess function to all the examples of iterable glue dataset
        raw_datasets = raw_datasets.map(
            function=preprocess_glue,
            batched=True,
            load_from_cache_file=False if data_args.overwrite_cache else True,
            desc="Tokenize the dataset.",
        )

    logger.info("Split train dataset.")
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(indices=range(data_args.max_train_samples))

    logger.info("Split validation dataset.")
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset.")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(indices=range(data_args.max_eval_samples))

    logger.info("Split test dataset.")
    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset.")
        test_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))

    # log a few random samples from the train dataset
    if training_args.do_train:
        # choose k unique random elements from a population sequence or set
        for i in random.sample(population=range(len(train_dataset)), k=3):
            logger.info(f"Sample {i} of the train dataset: {train_dataset[i]}")

    # get a base metric class
    logger.info(f"Get a base metric class.")
    if data_args.task_name is not None:
        metric = load_metric(path="glue", config_name=data_args.task_name)
    else:
        metric = load_metric(path="accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, np.ndarray]:
        """Get a metric score corresponding to a specific task.

        Args:
            eval_pred (EvalPrediction): An evaluation output containing labels.
        """
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        if data_args.task_name is not None:
            # compute metric
            result = metric.compute(predictions=preds, references=eval_pred.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()

            return result

        elif is_regression:
            return {"mse": ((preds - eval_pred.label_ids) ** 2).mean().item()}

        else:
            return {"accuracy": (preds == eval_pred.label_ids).astype(np.float32).mean().item()}

    logger.info("Set a data collator.")
    if data_args.pad_to_max_length:
        # a simple default data collator
        data_collator = default_data_collator
    elif training_args.fp16:
        # data collator that will dynamically pad the inputs received
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # init trainer
    logger.info("Initialize trainer.")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train
    logger.info("Start to train.")
    if training_args.do_train:
        # set a path for checkpoint if it exists
        ckpt = None
        if training_args.resume_from_checkpoint is not None:
            ckpt = training_args.resume_from_checkpoint
        elif last_ckpt is not None:
            ckpt = last_ckpt

        train_result = trainer.train(resume_from_checkpoint=ckpt)
        # get training result as metric
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        logger.info("Save model, tokenizer, and trainer state.")
        trainer.save_model()
        trainer.save_state()

        logger.info("Show and save training metrics.")
        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)

    # evaluate
    logger.info("Start to evaluate.")
    if training_args.do_eval:
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_matched"])

        for eval_dataset, _ in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            logger.info("Show and save evaluation metrics.")
            trainer.log_metrics(split="eval", metrics=metrics)
            trainer.save_metrics(split="eval", metrics=metrics)

    # test
    logger.info("Start to test.")
    if training_args.do_predict:
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(raw_datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            test_dataset = test_dataset.remove_columns("label")
            predictions = trainer.predict(test_dataset=test_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            logger.info(f"Save test results.")
            output_predict_file = os.path.join(
                training_args.output_dir,
                f"test_results:{task}_{training_args.run_name}.txt",
            )
            if trainer.is_world_process_zero():
                with open(file=output_predict_file, mode="w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    logger.info("Finished.")


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = get_glue_logger(training_args)
    run_glue(model_args=model_args, data_args=data_args, training_args=training_args, logger=logger)
