from transformers import AutoModelForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from helpers.metric import compute_metrics_baseline
from helpers.utils import *
import os
from pathlib import Path
from config.config_parser import *
from evaluation.get_predictions import get_predictions
from datasets import load_dataset
import sklearn


def model_evaluation(checkpoint, train_dataset, test_dataset, dataset_name, results_path, dataset_split="test"):
    df_predictions = get_predictions(checkpoint, results_path, test_dataset, dataset_split)

    if checkpoint.split('/')[-1].startswith('wordbert'):
        sentence = df_predictions.sentence.values
        word = df_predictions.word.values
        label = np.array([l[0] for l in df_predictions.label.values])
        prediction = df_predictions.prediction.values

        words_statistics, metaphor_words, non_metaphor_words = calculate_words_statistics(train_dataset, dataset_name,
                                                                                          per_word=True)
        real_eval = full_eval(data=[sentence, word], predictions=prediction, labels=label,
                              metaphor_words=metaphor_words, non_metaphor_words=non_metaphor_words, per_word=True)

    else:
        data = df_predictions.data.values
        predictions = df_predictions.predictions.values
        labels = df_predictions.labels.values
        words_statistics, metaphor_words, non_metaphor_words = calculate_words_statistics(train_dataset, dataset_name)
        real_eval = full_eval(data=data, predictions=predictions, labels=labels,
                          metaphor_words=metaphor_words, non_metaphor_words=non_metaphor_words)

    return real_eval

