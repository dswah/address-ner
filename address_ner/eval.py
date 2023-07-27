from functools import partial
from pathlib import Path

import click
import datasets
from datasets import ClassLabel, Sequence
import transformers
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from address_ner.logger import get_logger
from address_ner.utils import metric, task, compute_metrics, compute_metrics_summary, evaluate, tokenize_and_align_labels


MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"


@click.option("--model-dir", default=Path(__file__).parent.parent.joinpath("artifacts/"), type=click.Path())
@click.option("--dataset-dir", default=Path(__file__).parent.parent.joinpath("data/output/"), type=click.Path())
@click.option("--model-checkpoint", default=MODEL_CHECKPOINT, type=str)
@click.option("--batch-size", default=16, type=int)
@click.option("--learning-rate", default=2e-5, type=float)
@click.option("--num-train-epochs", "--epochs", default=3, type=int)
@click.option("--weight-decay", default=0.01, type=float)
@click.option("--label-all-tokens", default=True, type=bool)
@click.option("--downsample", default=10, type=int)
def run(
    model_dir: str,
    dataset_dir:str,
    model_checkpoint:str,
    batch_size:int,
    learning_rate:float,
    num_train_epochs:int,
    weight_decay:float,
    label_all_tokens:bool,
    downsample:int
):
    logger = get_logger(__name__)

    model_name = model_checkpoint.split("/")[-1]
    model_dir = str(Path(model_dir).joinpath(f"{model_name}-finetuned-{task}"))

    logger.info(f"Loading Dataset from: `{dataset_dir}`")
    dataset = datasets.load_from_disk(str(dataset_dir))

    label_list = dataset["train"].features[f"{task}_tags"].feature.names
    id2label = {i:label for i, label in enumerate(label_list)}
    label2id = {v:k for k, v in id2label.items()}

    logger.info(f"Loading Tokenizer and Model from: `{model_dir}`")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    model.config.id2label = id2label
    model.config.label2id = label2id

    logger.info(f"Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label_all_tokens":label_all_tokens})

    args = TrainingArguments(
        model_dir,
        evaluation_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        report_to="none"
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"].shard(downsample, 0),
        eval_dataset=tokenized_datasets["validation"].shard(downsample, 0),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics_summary, label_list=label_list)
    )

    logger.info(f"Computing Validation metrics...")
    val_metrics = evaluate(
        trainer,
        tokenized_datasets["validation"].shard(downsample,0),
        label_list
    )
    logger.info(f"Validation metrics\n\t{val_metrics}")

    logger.info(f"Computing Test metrics...")
    test_metrics = evaluate(
        trainer,
        tokenized_datasets["test"].shard(downsample,0),
        label_list
    )
    logger.info(f"Test metrics\n\t{test_metrics}")

    metrics = {}
    for metric in val_metrics:
        metrics[f"validation_{metric}"] = val_metrics[metric]
    for metric in test_metrics:
        metrics[f"test_{metric}"] = test_metrics[metric]

    parameters = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "downsample": downsample,
        "model_checkpoint": model_checkpoint,
        "label_all_tokens": label_all_tokens
    }

    # logger.info(f"Logging to MLFlow...")
    # mlflow_logger.log(
    #     model=model,
    #     tokenizer=tokenizer,
    #     metrics=metrics,
    #     parameters=parameters
    # )


if __name__ == "__main__":
    click.command()(run)()
