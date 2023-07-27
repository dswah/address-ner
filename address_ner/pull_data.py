import random
from pathlib import Path

import click
import datasets
import numpy as np

from address_ner.logger import get_logger

RANDOM_STATE = 42
BUFFER_SIZE = 1000

@click.option("--output-dir", "--output", default=Path(__file__).parent.parent.joinpath("data/input/"), type=click.Path())
@click.option("--train-size", default=5000, type=int)
@click.option("--val-size", "--validation-size", default=1000, type=int)
@click.option("--test-size", default=1000, type=int)
@click.option("--random-state", default=RANDOM_STATE, type=int)
@click.option("--buffer-size", default=BUFFER_SIZE, type=int)
def run(
    output_dir: str,
    train_size: int,
    val_size: int,
    test_size: int,
    random_state: int,
    buffer_size: int,
):
    logger = get_logger(__name__)

    logger.info(f"Using Random Seed {random_state}...")
    random.seed(random_state)
    np.random.seed(random_state)

    logger.info("Loading Raw Dataset...")
    dataset = datasets.load_dataset(
        "oscar",
        "unshuffled_deduplicated_de",
        split="train",
        streaming=True
    ).select_columns("text")

    dataset = dataset.shuffle(seed=random_state, buffer_size=buffer_size)

    def iterable_dataset_to_eager_dataset(size, offset=0, dataset=None):
        for sample in dataset.skip(offset).take(size):
            yield sample

    train = datasets.Dataset.from_generator(
        iterable_dataset_to_eager_dataset,
        gen_kwargs={
            "size": train_size,
            "offset": 0,
            "dataset": dataset
        }
    )

    val = datasets.Dataset.from_generator(
        iterable_dataset_to_eager_dataset,
        gen_kwargs={
            "size": val_size,
            "offset": train_size,
            "dataset": dataset
        })

    test = datasets.Dataset.from_generator(
        iterable_dataset_to_eager_dataset,
        gen_kwargs={
            "size": test_size,
            "offset": train_size + val_size,
            "dataset": dataset
        })

    logger.info(f"Writing Raw Dataset to {output_dir}")
    dataset = datasets.DatasetDict({
        "train": train,
        "validation": val,
        "test": test
    })
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    click.command()(run)()
