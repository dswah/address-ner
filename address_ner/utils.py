import re
from copy import deepcopy

from datasets import load_metric
import numpy as np


def _squeeze_entity(ent, text=None):
    """Strip preceeding and trailing whitespace from an entity

    Parameters
    ----------
    ent : dict containing
        "begin" -> int
        "end" -> int

    text : str or None
        if None, then the ent dict must also contain
            "text" -> str

    Returns
    -------
    dict
    """
    try:
        ent_text = ent.get("text") or text[ent["begin"] : ent["end"]]
    except TypeError:
        raise ValueError(f"cannot squeeze entity: `{ent}` without text.")

    # squeeze left side
    match = re.search(r"^\s+", ent_text)
    if match:
        ent["begin"] += match.span()[1]
        ent_text = ent_text[match.span()[1] :]

    # squeeze right side
    match = re.search(r"\s+$", ent_text)
    if match:
        ent["end"] = ent["begin"] + match.span()[0]
        ent_text = ent_text[: match.span()[0]]

    ent["text"] = ent_text
    return ent


def squeeze_entities(entities, text=None):
    """Strip preceeding and trailing whitespace from entities

    Parameters
    ----------
    entities : list of dicts, each containing
        "begin" -> int
        "end" -> int

    text : str or None
        if None, then the entity dicts must also contain
            "text" -> str

    Returns
    -------
    list of dict
    """
    return [_squeeze_entity(ent, text=text) for ent in entities]


def resolve_overlap(entities):
    """greedily remove overlapping entities, keeping the longest ones

    Parameters
    ----------
    entities : list of dicts, each containing
        "begin" -> int
        "end" -> int

    Returns
    -------
    list of dict
    """
    "greedily keep longest entities, return sorted by offset begin"
    entities = sorted(entities, key=lambda ent: ent["end"] - ent["begin"], reverse=True)
    used = set()
    keep = []
    for ent in entities:
        idxs = range(ent["begin"], ent["end"])
        if not used.intersection(idxs):
            used = used.union(idxs)
            keep.append(ent)
    return sorted(keep, key=lambda ent: ent["begin"])


def relative_to_absolute(relative_entities, entity):
    """adjust relative offsets of entities to global offsets

    Parameters
    ----------
    relative_entities : list of dict
    entity : dict

    Returns
    -------
    global_entities : list of dict
    """
    absolute_entities = []
    for ent in relative_entities:
        ent["begin"] += entity["begin"]
        ent["end"] += entity["begin"]
        absolute_entities.append(ent)
    return absolute_entities


def align_entities(ents, doc):
    """align entity dicts to the tokens in a spacy Doc

    Parameters
    ----------
    ents : list of entity dicts
    doc: spacy Doc

    Returns
    -------
    list of entity dicts
    """
    ents = deepcopy(ents)

    # build map for all BEGIN offsets
    begin_map = {}
    start = 0
    for tok in doc:
        if tok.is_space:
            continue

        begin, end = tok.idx, tok.idx + len(tok)

        begin_map.update({k:begin for k in range(start, end)})
        start = end

        # ensure that valid token begins map to themselves
        begin_map[begin] = begin

    # build map for all END offsets
    end_map = {}
    stop = len(doc.text) + 1
    for tok in list(doc)[::-1]:
        if tok.is_space:
            continue

        begin, end = tok.idx, tok.idx + len(tok)

        end_map.update({k:end for k in range(begin, stop)})
        stop = begin

        # ensure that valid token ends map to themselves
        end_map[end] = end

    # apply maps to all entity offsets
    for ent in ents:
        # ensure we can map
        ent["begin"] = max(0, ent["begin"])
        ent["end"] = min(len(doc.text) + 1, ent["end"])

        # apply map
        ent["begin"] = begin_map[ent["begin"]]
        ent["end"] = end_map[ent["end"]]
        ent["text"] = doc.text[ent["begin"]:ent["end"]]

    return ents



task = "ner" # Should be one of "ner", "pos" or "chunk"
metric = load_metric("seqeval")


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """from huggingface docs
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, label_list):
    """from huggingface docs
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    test_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    test_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=test_predictions, references=test_labels)
    return results


def compute_metrics_summary(p, label_list):
    results = compute_metrics(p, label_list)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def evaluate(trainer, test_data, label_list):
    predictions, labels, _ = trainer.predict(test_data)
    return compute_metrics_summary((predictions, labels), label_list)
