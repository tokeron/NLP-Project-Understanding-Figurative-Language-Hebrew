import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    Set random seeds for reproducibility
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_args(training_args):
    """
    This is where you can check the validity of the configuration and set extra attributes that can't be incorporated
    in the YAML file
    """
    return training_args


def align_labels_with_tokens(labels, word_ids, ignore_subtokens=False):
    """
    # Align the labels to the words after tokenization
    :param labels: Original labels
    :param word_ids: List of word ids after tokenization
    :param ignore_subtokens: if True, the labels are aligned to the first sub-word and the rest are ignored
    :return: new labels aligned to the words after tokenization
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX or -100 (depending on the flag ignore_subtokens)
            if label % 2 == 1:
                if ignore_subtokens:
                    label = -100
                else:
                    label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(dataset, tokenizer, ignore_subtokens=False):
    """
    Tokenize the dataset and align the labels to the words after tokenization
    :param dataset: dataset to tokenize
    :param tokenizer: tokenizer to use for tokenization
    :param ignore_subtokens:  if True, the labels are aligned to the first sub-word and the rest are ignored
    :return: tokenized dataset with aligned labels
    """
    tokenized_inputs = tokenizer(
        dataset["data"], truncation=True, is_split_into_words=True
    )
    all_labels = dataset["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, ignore_subtokens))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def get_actual_predictions(predictions, tokenized_dataset, tokenizer):
    """
    Get the actual predictions.
    The current method is to take the maximal value of the sub-words of a word as the prediction of the word
    :param predictions: list of predictions on sub-words
    :param tokenized_dataset: list of tokenized dataset
    :param tokenizer: tokenizer used for tokenization
    :return: predictions on original words
    """
    actual_predictions = []
    for i, (prediction_array, input_ids) in enumerate(zip(predictions.predictions, tokenized_dataset['input_ids'])):
        current_predictions = []
        prediction_array = prediction_array.argmax(-1)
        words = tokenizer.convert_ids_to_tokens(input_ids)
        for word_index, (word, prediction) in enumerate(zip(words, prediction_array)):
            if word != "[SEP]" and word != "[CLS]" and word.find("#") == -1:
                while word_index+1 < len(input_ids) and words[word_index+1].find("#") != -1:
                    prediction = max(prediction, prediction_array[word_index+1])
                    word_index += 1
                current_predictions.append(prediction)
        actual_predictions.append(current_predictions)
    return actual_predictions


def save_heatmap(data, labels, predictions, filename):
    """
    Prints the results of the model on the test set
    :param data: the data
    :param labels: the labels
    :param predictions: the predictions
    :param filename: the filename to save the results
    :return:
    """
    confusion_matrix = np.zeros(([2, 2]))
    false_positives_words = []
    false_negatives_words = []
    for data_index, (data_item, labels_item, predictions_item) in enumerate(zip(data, labels, predictions)):
        for word_index, (word, label, prediction) in enumerate(zip(data_item, labels_item, predictions_item)):
            if label != 0 and prediction == 0:
                confusion_matrix[0, 1] += 1
                false_negatives_words.append(word)
            elif label == 0 and prediction != 0:
                confusion_matrix[1, 0] += 1
                false_positives_words.append(word)
            elif label == 0 and prediction == 0:
                confusion_matrix[0, 0] += 1
            elif label != 0 and prediction != 0:
                confusion_matrix[1, 1] += 1

    # Plot the confusion matrix as a heatmap
    ax = sns.heatmap(confusion_matrix, annot=True)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig = ax.get_figure()
    fig.savefig(filename + "_heatmap.png")
