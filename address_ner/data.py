import re
import random
from joblib import Parallel, delayed
from itertools import chain
from collections import defaultdict
from typing import List
from functools import partial
from pathlib import Path

import click
import datasets
from datasets import Sequence, ClassLabel
import numpy as np
import spacy
from spacy.training import offsets_to_biluo_tags, biluo_tags_to_spans
from conny_ml.extraction.entities.utils import resolve_overlap, align_entities

from address_ner.address_generator import AddressGenerator
from address_ner.logger import get_logger


PATTERNS = [
    "\w+berg",
    "allee",
    "str.",
    "strasse",
    "straße",
    "(?<!\d\d\d.)\d\d\d\d\d", # non telephone zip code
    "paul-lincke-ufer",
    "8c",
    "LINCKE-UFER",
    "neukölln",
    "damm",
    "germany",
    "berlin",
    "münchen",
    "munchen",
    "stuttgart",
    "deutschland",
    "potsdam",
    "bremen",
    "wiesbaden",
    "hanover",
    "schwerin",
    "düsseldorf",
    "dusseldorf",
    "mainz",
    "saarbrücken",
    "saarbrucken",
    "dresden",
    "magdeburg",
    "kiel",
    "erfunt",
    "postfach", # PO box,
    "adresse"
]

PATTERNS = re.compile(r"|".join(PATTERNS), flags=re.IGNORECASE)


def _has_address(text):
    return bool(PATTERNS.search(text))


def _sample_n_characters(character_distribution=(50,5), minimum=10):
    return max(int(random.gauss(*character_distribution)), minimum)


def make_samples(text, character_distribution=(50, 5), generator=None, random_state=42):
    """
    process:
        - split text by \n
        - remove lines that are "address-like"
        - collect lines into lists until length is just longer than a sampled character limit
        - generate an address for each list
        - insert address uniformly at random each list
        - randomly join each list of lines with \n and \s
    """
    random.seed(random_state)

    lines = text.split("\n")
    lines = [line for line in lines if not _has_address(line)]

    if len(lines) == 0:
        return []

    # split into reasonable sum-samples
    samples = []
    sample = []
    length = _sample_n_characters(character_distribution)
    while lines:
        sample.append(lines.pop(0))

        if np.sum([len(line) for line in sample]) >= length:
            samples.append(sample)
            sample = []
            length = _sample_n_characters(character_distribution)

    # collect last sample regardless of length
    if sample:
        samples.append(sample)

    # generate addresses and insert uniformly at random
    entities = []
    for sample in samples:
        address = generator()
        index = random.randint(0, len(sample))
        sample.insert(index, address)

        entity = {}
        entity["text"] = address
        entity["begin"] = int(np.sum([len(line) for line in sample[:index]]) + len(sample[:index])) # lens of strings + joins
        entity["end"] = int(entity["begin"] + len(entity["text"]))
        entity["label"] = "address"

        entities.append(entity)

    # randomly join sample text with " " or "\n"
    for i, sample in enumerate(samples):

        sample_text = ""
        while len(sample) > 1:
            sample_text += sample.pop(0)
            sample_text += random.choice([" ", "\n"])
        sample_text += sample.pop() # add last string

        samples[i] = sample_text


    # sanity check
    for ent, sample in zip(entities, samples):
        assert sample[ent["begin"]:ent["end"]] == ent["text"]

    return list(zip(samples, entities))


def _get_spacy_model(model="de_core_news_sm"):
    exclude = [
        'tok2vec',
        'tagger',
        'morphologizer',
        'parser',
        'attribute_ruler',
        'lemmatizer',
        'ner'
    ]

    try:
        nlp = spacy.load(model, exclude=exclude)
    except OSError as error:
        import subprocess

        command = f'poetry run python -m spacy download {model}'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    nlp = spacy.load(model, exclude=exclude)
    return nlp


def sample_to_conll(sample, nlp=None, scheme="IOB", output_key="ner_tags"):
    assert scheme in ["IOB", "BILUO"]

    nlp = nlp or _get_spacy_model(model="de_core_news_sm")
    doc = nlp(sample[0])
    doc_ents = align_entities([sample[1]], doc)
    doc_ents = resolve_overlap(doc_ents)

    tags = offsets_to_biluo_tags(doc, [(e["begin"], e["end"], e["label"]) for e in doc_ents])
    if scheme == "IOB":
        for i, tag in enumerate(tags):
            if tag[0] == "U":
                tags[i] = "B" + str(tag)[1:]
            if tag[0] == "L":
                tags[i] = "I" + str(tag)[1:]

    toks = [str(tok) for tok in doc]

    return {"tokens": toks, output_key: tags}


def _batch_to_conll(batch, scheme="IOB", output_key="ner_tags"):
    nlp = _get_spacy_model(model="de_core_news_sm")

    results = []
    for datum in batch:
        results.append(sample_to_conll(datum, nlp=nlp, scheme=scheme, output_key=output_key))
    return results


def _flatten_dicts(dicts):
    """convert list of dicts to dict of lists for loading into HF dataset"""
    out = defaultdict(list)
    keys = list(set(chain(*(tuple(d.keys()) for d in dicts))))
    for d in dicts:
        for k in keys:
            out[k].append(d.get(k))
    return dict(out)


def _get_labels(dataset, feature_name="ner_tags"):
    labels = list(set(chain(*dataset.map(lambda sample: {"labels": list(set(sample[feature_name]))})["labels"])))
    labels.remove("O")
    labels.sort(key=lambda label: label[::-1])
    labels = ["O"] + labels
    return labels


def _make_dataset(
    dataset_path: str,
    address_generator: AddressGenerator,
    character_distribution: List[int],
    split: str,
    labels: List[str]=None,
    n_workers: int=8,
    limit: int=None,
    feature_name: str="ner_tags",
    logger=None
):

    if logger:
        logger.info(f"Gathering {split.upper()} data from {dataset_path}")

    if limit is not None:
        if logger:
            logger.info(f"Limiting {split.upper()} split to {limit} samples")

        dataset = datasets.load_dataset(
            "json",
            data_files={split:str(dataset_path)},
            split=f'{split}[:{limit}]'
        )
    else:
        dataset = datasets.load_dataset(
            "json",
            data_files={split:str(dataset_path)},
        )

    # generate addresses for each text
    dataset = Parallel(n_jobs=n_workers)(delayed(make_samples)(
        datum["text"],
        character_distribution,
        address_generator.sample) for datum in dataset)

    # flatten into list of objects
    dataset = list(chain(*dataset))

    # batch and convert to conll
    dataset = np.array_split(dataset, n_workers)
    _batched_fn = partial(_batch_to_conll, output_key=feature_name)
    dataset = Parallel(n_jobs=n_workers)(delayed(_batched_fn)(batch) for batch in dataset)

    # flatten into list of objects
    dataset = list(chain(*dataset))

    # convert to huggingface dataset
    dataset = datasets.Dataset.from_dict(_flatten_dicts(dataset))

    # use integers instead of strings for class names
    labels = labels or _get_labels(dataset, feature_name=feature_name)
    dataset.features[feature_name] = Sequence(ClassLabel(num_classes=len(labels), names=labels))
    tag2id = {k:i for i, k in enumerate(labels)}
    dataset = dataset.map(lambda sample: {feature_name: [tag2id[tag] for tag in sample[feature_name]]})

    if logger:
        logger.info(f"{split.upper()} split has {len(dataset)} samples")

    return dataset


@click.option("--output-dir", default=Path(__file__).parent.parent.joinpath("data/output/"), type=click.Path())
@click.option("--train-path", default=Path(__file__).parent.parent.joinpath("data/input/train.jsonl"), type=click.Path())
@click.option("--val-path", "--validation-path", default=Path(__file__).parent.parent.joinpath("data/input/validation.jsonl"), type=click.Path())
@click.option("--test-path", default=Path(__file__).parent.parent.joinpath("data/input/test.jsonl"), type=click.Path())
@click.option("--limit", default=None, type=int)
@click.option("--char-mean", default=50, type=int)
@click.option("--char-std", default=5, type=int)
@click.option("--feature-name", default="ner_tags", type=str)
@click.option("--n-workers", default=8, type=int)
@click.option("--random-state", default=42, type=int)
def run(
    output_dir: str,
    train_path: str,
    val_path: str,
    test_path: str,
    limit: int,
    char_mean: int,
    char_std: int,
    feature_name: str,
    n_workers: int,
    random_state: int,
):
    logger = get_logger(__name__)

    logger.info(f"Using Random Seed {random_state}...")
    random.seed(random_state)
    np.random.seed(random_state)

    address_generator = AddressGenerator(random_state=random_state)

    logger.info("Loading Datasets...")
    train = _make_dataset(train_path, address_generator, (char_mean, char_std), split="train", feature_name=feature_name, n_workers=n_workers, limit=limit, logger=logger)
    labels = train.features[feature_name].feature.names

    val = _make_dataset(val_path, address_generator, (char_mean, char_std), split="val", feature_name=feature_name, labels=labels, n_workers=n_workers, limit=limit, logger=logger)
    test = _make_dataset(test_path, address_generator, (char_mean, char_std), split="test", feature_name=feature_name, labels=labels, n_workers=n_workers, limit=limit, logger=logger)

    logger.info(f"Writing Dataset to {output_dir}")
    dataset = datasets.DatasetDict({
        "train": train,
        "validation": val,
        "test": test
    })
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    click.command()(run)()
