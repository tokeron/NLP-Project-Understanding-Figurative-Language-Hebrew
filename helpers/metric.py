import numpy as np
from datasets import load_metric
import sklearn.metrics as metrics
import torch
from transformers import T5TokenizerFast
from config.config_parser import *
import pandas as pd
label_names = ["O", "B-metaphor", "B-metaphor"]


# This function computes the overall score for a batch of predictions and labels
def compute_metrics_baseline(eval_preds):
    metric = load_metric("seqeval")
    # logits, labels = eval_preds
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    # TODO commented out for now
    # if len(logits) == 2:
    #     logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # true_logits = [logit for logit, label in zip(logits, labels) if label != -100]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels,
                                 zero_division=0)

    true_labels = [label for sublist in true_labels for label in sublist]
    true_predictions = [label for sublist in true_predictions for label in sublist]

    binary_labels = [1 if l == "B-metaphor" else 0 for l in true_labels]
    binary_predictions = [1 if p == "B-metaphor" else 0 for p in true_predictions]

    if sum(binary_labels) == 0 and sum(binary_predictions) == 0:
        results = {
            "precision": 1,
            "recall": 1,
            "f1": 1,
            "accuracy": 1,
        }
    else:
        results = {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
            # "sklearn_precision": metrics.precision_score(binary_labels, binary_predictions),
            # "sklearn_recall": metrics.recall_score(binary_labels, binary_predictions),
            # "sklearn_f1": metrics.f1_score(binary_labels, binary_predictions),
            # "sklearn_f1_macro": metrics.f1_score(binary_labels, binary_predictions, average="macro"),
            # "sklearn_f1_weighted": metrics.f1_score(binary_labels, binary_predictions, average="weighted"),
            # "sklearn_accuracy": metrics.accuracy_score(binary_labels, binary_predictions),
        }

    return results


def compute_metrics_melbert(eval_preds: object) -> object:
    metric = load_metric("seqeval")

    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    if len(logits) == 2:
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    labels = labels.transpose(0, 1).reshape(-1)
    true_labels, true_predictions, true_logits = [], [], []
    binary_labels, binary_predictions = [], []
    for prediction, label in zip(predictions, labels):
        if label != -100:
            true_labels.append(label_names[label])
            true_predictions.append(label_names[prediction])
            true_logits.append(logits[prediction])
            binary_labels.append(1 if label == 1 or label == 2 else 0)
            binary_predictions.append(1 if prediction == 1 or prediction == 2 else 0)

    all_metrics = metric.compute(predictions=[true_predictions], references=[true_labels],
                                 zero_division=0)


    results = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
        # "sklearn_precision": metrics.precision_score(binary_labels, binary_predictions),
        # "sklearn_recall": metrics.recall_score(binary_labels, binary_predictions),
        # "sklearn_f1": metrics.f1_score(binary_labels, binary_predictions),
        # "sklearn_f1_macro": metrics.f1_score(binary_labels, binary_predictions, average="macro"),
        # "sklearn_f1_weighted": metrics.f1_score(binary_labels, binary_predictions, average="weighted"),
        # "sklearn_accuracy": metrics.accuracy_score(binary_labels, binary_predictions),
    }

    return results
