import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from wordcloud import WordCloud, STOPWORDS
import os
import string
import json
import jsonlines
import pickle
from datetime import datetime

# import metrics from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from transformers import MT5Model, T5TokenizerFast, MT5EncoderModel
from BEREL_PyTorch_Model.rabtokenizer import RabbinicTokenizer
from transformers import BertTokenizer, BertForMaskedLM
from transformers import convert_slow_tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer, BertConfig, BertTokenizerFast

from helpers.metric import *

import datasets

label_names = ["O", "B-metaphor", "I-metaphor"]


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
            if label == 1:
                if ignore_subtokens:
                    label = -100
                else:
                    label += 1
            if label == 2:
                if ignore_subtokens:
                    label = -100

            new_labels.append(label)
    if len(new_labels) != len(word_ids):
        raise ValueError(
            "The number of labels does not match the number of tokens in the sentence"
        )
    return new_labels


def tokenize(dataset, tokenizer, ignore_subtokens=False):
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
    tokenized_inputs["labels"] = dataset["labels"]
    return tokenized_inputs


def tokenize_sentence_word(dataset, tokenizer, ignore_subtokens=False):
    """
    Tokenize the dataset and align the labels to the words after tokenization
    :param dataset: dataset to tokenize
    :param tokenizer: tokenizer to use for tokenization
    :param ignore_subtokens:  if True, the labels are aligned to the first sub-word and the rest are ignored
    :return: tokenized dataset with aligned labels
    """
    tokenized_sentence = tokenizer(
        dataset["sentence"], truncation=True, is_split_into_words=True, padding=True
    )

    tokenized_word = tokenizer(
        dataset["word"], truncation=True, is_split_into_words=True, padding=True
    )

    tokenized_input = tokenized_sentence
    tokenized_input["word_input_ids"] = tokenized_word.input_ids
    tokenized_input["word_attention_mask"] = tokenized_word.attention_mask
    tokenized_input["word_token_type_ids"] = tokenized_word.token_type_ids
    tokenized_input["word_idx"] = dataset["word_idx"]
    tokenized_input["labels"] = dataset["label"]
    return tokenized_input


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
    # tokenized_inputs["word_ids"] = tokenized_inputs.word_ids
    return tokenized_inputs


def get_actual_predictions(predictions, tokenized_dataset, tokenizer, raw_dataset, per_word=False):
    """
    Get the actual predictions.
    The current method is to take the maximal value of the sub-words of a word as the prediction of the word
    :param predictions: list of predictions on sub-words
    :param tokenized_dataset: list of tokenized dataset
    :param tokenizer: tokenizer used for tokenization
    :return: predictions on original words
    """
    if per_word:
        actual_predictions = predictions.predictions.argmax(-1).T[0]
        return actual_predictions
    else:
        frequent_predictions = []
        max_predictions = []
        min_predictions = []
        first_predictions = []
        last_predictions = []

        tokenized_inputs = tokenizer(
            raw_dataset["data"], truncation=True, is_split_into_words=True
        )
        actual_predictions = []
        predictions = predictions.predictions
        if len(predictions) == 2:
            predictions = predictions[0]
        for i, (prediction_array, input_ids) in enumerate(zip(predictions, tokenized_dataset['input_ids'])):
            word_ids = tokenized_inputs.word_ids(i)
            current_sentence = []
            current_sentence_frequent = []
            current_sentence_max = []
            current_sentence_min = []
            current_sentence_first = []
            current_sentence_last = []

            prediction_array = prediction_array.argmax(-1)
            words = tokenizer.convert_ids_to_tokens(input_ids)
            for word_index in range(len(raw_dataset["labels"][i])):
                # get indices of word_index in word_ids array
                indices = np.where(np.array(word_ids) == word_index)[0]

                # get the label of the words
                curr_predictions = prediction_array[indices]

                # get the maximal value of the sub-words
                if len(curr_predictions) > 0:
                    max_value = np.max(curr_predictions)
                else:
                    max_value = 0
                    print("No predictions for word {}".format(raw_dataset["data"][i][word_index]))

                # # get the minimal value of the sub-words
                # min_value = np.min(curr_predictions)
                #
                # # get the first value of the sub-words
                # first_value = curr_predictions[0]
                #
                # # get the last value of the sub-words
                # last_value = curr_predictions[-1]
                #
                # # get the most frequent label of the sub-words
                # frequent_value = np.argmax(np.bincount(prediction_array[indices]))

                current_sentence.append(max_value)
                # current_sentence_frequent.append(frequent_value)
                # current_sentence_max.append(max_value)
                # current_sentence_min.append(min_value)
                # current_sentence_first.append(first_value)
                # current_sentence_last.append(last_value)

            if len(current_sentence) != len(raw_dataset["labels"][i]):
                # Temporal solution to avoid errors
                # current_sentence = current_sentence[:len(raw_dataset["labels"][i])]
                raise ValueError("Predictions are not of the same length as the original data")
            actual_predictions.append(current_sentence)
            # frequent_predictions.append(current_sentence_frequent)
            # max_predictions.append(current_sentence_max)
            # min_predictions.append(current_sentence_min)
            # first_predictions.append(current_sentence_first)
            # last_predictions.append(current_sentence_last)

    # full_eval_freq = full_eval(raw_dataset["data"], raw_dataset["labels"], frequent_predictions, metaphor_words=[])
    # full_eval_max = full_eval(raw_dataset["data"], raw_dataset["labels"], max_predictions, metaphor_words=[])
    # full_eval_min = full_eval(raw_dataset["data"], raw_dataset["labels"], min_predictions, metaphor_words=[])
    # full_eval_first = full_eval(raw_dataset["data"], raw_dataset["labels"], first_predictions, metaphor_words=[])
    # full_eval_last = full_eval(raw_dataset["data"], raw_dataset["labels"], last_predictions, metaphor_words=[])
    #
    # print("Frequent: f1: {}, acc: {}".format(full_eval_freq["total"]["f1"], full_eval_freq["total"]["accuracy"]))
    # print("Max: f1: {}, acc: {}".format(full_eval_max["total"]["f1"], full_eval_max["total"]["accuracy"]))
    # print("Min: f1: {}, acc: {}".format(full_eval_min["total"]["f1"], full_eval_min["total"]["accuracy"]))
    # print("First: f1: {}, acc: {}".format(full_eval_first["total"]["f1"], full_eval_first["total"]["accuracy"]))
    # print("Last: f1: {}, acc: {}".format(full_eval_last["total"]["f1"], full_eval_last["total"]["accuracy"]))

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


def plot_word_cloud(words, filename, title):
    # reverse text ( end to start because this is hebrew )
    if len(words) == 0:
        return
    bidi_text = get_display(words)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          font_path='../Font/FreeSans/FreeSansBold.ttf',
                          min_font_size=10).generate(bidi_text)
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title)
    plt.show()
    fig = plt.gcf()
    fig.savefig(filename)


def count_words(train_datasets):
    metaphor_words, non_metaphor_words = {}, {}
    for tuple in train_datasets:
        for word, label in zip(tuple["data"], tuple["labels"]):
            if label != 0:
                if word in metaphor_words:
                    metaphor_words[word] += 1
                else:
                    metaphor_words[word] = 1
            else:
                if word in non_metaphor_words:
                    non_metaphor_words[word] += 1
                else:
                    non_metaphor_words[word] = 1
    return metaphor_words, non_metaphor_words


def create_word_clouds(seen_TP, seen_TN, seen_FP, seen_FN, unseen_TP, unseen_TN, unseen_FP, unseen_FN, path, test_name):
    plot_word_cloud(seen_TP, os.path.join(path, test_name + "_seen_TP_wordcloud.png"), title="Seen True Positives")
    plot_word_cloud(seen_FP, os.path.join(path, test_name + "_seen_FP_wordcloud.png"), title="Seen False Positives")
    plot_word_cloud(seen_FN, os.path.join(path, test_name + "_seen_FN_wordcloud.png"), title="Seen False Negatives")
    plot_word_cloud(seen_TN, os.path.join(path, test_name + "_seen_TN_wordcloud.png"), title="Seen True Negatives")
    plot_word_cloud(unseen_TP, os.path.join(path, test_name + "_unseen_TP_wordcloud.png"), title="Unseen True Positives")
    plot_word_cloud(unseen_FP, os.path.join(path, test_name + "_unseen_FP_wordcloud.png"), title="Unseen False Positives")
    plot_word_cloud(unseen_FN, os.path.join(path, test_name + "_unseen_FN_wordcloud.png"), title="Unseen False Negatives")
    plot_word_cloud(unseen_TN, os.path.join(path, test_name + "_unseen_TN_wordcloud.png"), title="Unseen True Negatives")


def calculate_words_statistics(raw_dataset, dataset_name, per_word=False):
    # # if the calculation is done before, just load the pickle file
    # path = '/home/tok/figurative-language/data/words_statistics_{}_updated_02.pkl'.format(dataset_name.split('.')[0].split('/')[-1])
    # if os.path.exists(path):
    #     with open(path, 'rb') as f:
    #         words_statistics, metaphor_words, non_metaphor_words = pickle.load(f)
    # else:
    metaphor_words = []
    non_metaphor_words = []
    words_statistics = {}
    if per_word:
        for i, example in enumerate(raw_dataset):
            word = example["word"][0]
            label = example["label"][0]
            if word not in words_statistics:
                if label == 0:
                    words_statistics[word] = {'metaphor': 0, 'non_metaphor': 1}
                    non_metaphor_words.append(word)
                else:  # It's a metaphor
                    words_statistics[word] = {'metaphor': 1, 'non_metaphor': 0}
                    metaphor_words.append(word)
            else:
                if label == 0:
                    words_statistics[word]['non_metaphor'] += 1
                    non_metaphor_words.append(word)
                else:  # It's a metaphor
                    words_statistics[word]['metaphor'] += 1
                    metaphor_words.append(word)

    else:
        for i, example in enumerate(raw_dataset):
            for word, label in zip(example['data'], example['labels']):
                if word not in words_statistics:
                    if label == 0:
                        words_statistics[word] = {'metaphor': 0, 'non_metaphor': 1}
                        non_metaphor_words.append(word)
                    else:  # It's a metaphor
                        words_statistics[word] = {'metaphor': 1, 'non_metaphor': 0}
                        metaphor_words.append(word)
                else:
                    if label == 0:
                        words_statistics[word]['non_metaphor'] += 1
                        non_metaphor_words.append(word)
                    else:  # It's a metaphor
                        words_statistics[word]['metaphor'] += 1
                        metaphor_words.append(word)
        # save pickle file for later use
        dataset_name = dataset_name.split('.')[0].split('/')[-1]
        # with open(path, 'wb') as f:
        #     pickle.dump([words_statistics, metaphor_words, non_metaphor_words], f)
    return words_statistics, metaphor_words, non_metaphor_words


def calculate_predictions_majority(dataset, words_statistics):
    predictions = []
    for i, example in enumerate(dataset):
        curr_pred = []
        for word, _ in zip(example['data'], example['labels']):
            if word not in words_statistics:
                curr_pred.append(0)
            else:
                if words_statistics[word]['metaphor'] > words_statistics[word]['non_metaphor']:
                    curr_pred.append(1)
                else:
                    curr_pred.append(0)
        predictions.append(curr_pred)
    return predictions


def full_eval(data, labels, predictions, metaphor_words, non_metaphor_words=None, per_word=False):
    # calculate metrics
    if per_word:
        labels_for_report = labels
        predictions_for_report = predictions
    else:
        labels_for_report = np.array([item for sublist in labels for item in sublist])
        predictions_for_report = np.array([item for sublist in predictions for item in sublist])

    # replace label 2 with 1 to make it compatible with the metrics
    labels_for_report[labels_for_report == 2] = 1
    predictions_for_report[predictions_for_report == 2] = 1
    accuracy = accuracy_score(labels_for_report, predictions_for_report)
    precision = precision_score(labels_for_report, predictions_for_report)
    recall = recall_score(labels_for_report, predictions_for_report)
    f1 = f1_score(labels_for_report, predictions_for_report)

    cm = confusion_matrix(labels_for_report, predictions_for_report)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    clasification_report = classification_report(labels_for_report, predictions_for_report)

    sklearn_results = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                       'cf_matrix': cm, 'classification_report': clasification_report}

    unseen_TP, unseen_FP, unseen_FN, unseen_TN = "", "", "", ""
    seen_TP, seen_FP, seen_FN, seen_TN = "", "", "", ""
    total_seen, total_unseen = 0, 0
    unseen_TP_count, unseen_FP_count, unseen_FN_count, unseen_TN_count = 0, 0, 0, 0
    seen_TP_count, seen_FP_count, seen_FN_count, seen_TN_count = 0, 0, 0, 0
    if per_word:
        sentences, words = data
        for data_index, (word, label, prediction) in enumerate(zip(words, labels, predictions)):
            word = word[0]
            if label != 0 and prediction == 0:
                if word in metaphor_words:
                    seen_FN += " " + word + " "
                    seen_FN_count += 1
                    total_seen += 1
                else:
                    unseen_FN += " " + word + " "
                    unseen_FN_count += 1
                    total_unseen += 1
            elif label == 0 and prediction != 0:
                if word in metaphor_words:
                    seen_FP += " " + word + " "
                    seen_FP_count += 1
                    total_seen += 1
                else:
                    unseen_FP += " " + word + " "
                    unseen_FP_count += 1
                    total_unseen += 1
            elif label == 0 and prediction == 0:
                if word in metaphor_words:
                    seen_TN += " " + word + " "
                    seen_TN_count += 1
                    total_seen += 1
                else:
                    unseen_TN += " " + word + " "
                    unseen_TN_count += 1
                    total_unseen += 1
            elif label != 0 and prediction != 0:
                if word in metaphor_words:
                    seen_TP += " " + word + " "
                    seen_TP_count += 1
                    total_seen += 1
                else:
                    unseen_TP += " " + word + " "
                    unseen_TP_count += 1
                    total_unseen += 1
    else:
        for data_index, (data_item, labels_item, predictions_item) in enumerate(zip(data, labels, predictions)):
            for word_index, (word, label, prediction) in enumerate(zip(data_item, labels_item, predictions_item)):
                if label != 0 and prediction == 0:
                    if word in metaphor_words:
                        seen_FN += " " + word + " "
                        seen_FN_count += 1
                        total_seen += 1
                    else:
                        unseen_FN += " " + word + " "
                        unseen_FN_count += 1
                        total_unseen += 1
                elif label == 0 and prediction != 0:
                    if word in metaphor_words:
                        seen_FP += " " + word + " "
                        seen_FP_count += 1
                        total_seen += 1
                    else:
                        unseen_FP += " " + word + " "
                        unseen_FP_count += 1
                        total_unseen += 1
                elif label == 0 and prediction == 0:
                    if word in metaphor_words:
                        seen_TN += " " + word + " "
                        seen_TN_count += 1
                        total_seen += 1
                    else:
                        unseen_TN += " " + word + " "
                        unseen_TN_count += 1
                        total_unseen += 1
                elif label != 0 and prediction != 0:
                    if word in metaphor_words:
                        seen_TP += " " + word + " "
                        seen_TP_count += 1
                        total_seen += 1
                    else:
                        unseen_TP += " " + word + " "
                        unseen_TP_count += 1
                        total_unseen += 1

    # create confusion matrix for all data
    total_TP_count = seen_TP_count + unseen_TP_count
    total_FP_count = seen_FP_count + unseen_FP_count
    total_FN_count = seen_FN_count + unseen_FN_count
    total_TN_count = seen_TN_count + unseen_TN_count
    cf_matrix_total = np.array([[total_TP_count, total_FP_count], [total_FN_count, total_TN_count]])
    cf_matrix_unseen = np.array([[unseen_TP_count, unseen_FP_count], [unseen_FN_count, unseen_TN_count]])
    cf_matrix_seen = np.array([[seen_TP_count, seen_FP_count], [seen_FN_count, seen_TN_count]])

    # Calculate accuracy, precision, recall, f1-score of seen words
    if total_seen == 0:
        accuracy_seen = 0
    else:
        accuracy_seen = (seen_TP_count + seen_TN_count) / total_seen
    if seen_TP_count + seen_FP_count == 0:
        precision_seen = 0
    else:
        precision_seen = seen_TP_count / (seen_TP_count + seen_FP_count)
    if seen_TP_count + seen_FN_count == 0:
        recall_seen = 0
    else:
        recall_seen = seen_TP_count / (seen_TP_count + seen_FN_count)
    if precision_seen + recall_seen == 0:
        f1_score_seen = 0
    else:
        f1_score_seen = 2 * (precision_seen * recall_seen) / (precision_seen + recall_seen)
    # Calculate accuracy, precision, recall, f1-score of unseen words
    accuracy_unseen = (unseen_TP_count + unseen_TN_count) / (total_unseen)
    if unseen_TP_count + unseen_FP_count == 0:
        precision_unseen = 0
    else:
        precision_unseen = unseen_TP_count / (unseen_TP_count + unseen_FP_count)
    if unseen_TP_count + unseen_FN_count == 0:
        recall_unseen = 0
    else:
        recall_unseen = unseen_TP_count / (unseen_TP_count + unseen_FN_count)
    if precision_unseen + recall_unseen == 0:
        f1_score_unseen = 0
    else:
        f1_score_unseen = 2 * (precision_unseen * recall_unseen) / (precision_unseen + recall_unseen)
    # Calculate accuracy, precision, recall, f1-score of all words
    if total_seen + total_unseen == 0:
        accuracy_total = 0
    else:
        accuracy_total = (total_TP_count + total_TN_count) / (total_seen + total_unseen)
    if total_TP_count + total_FP_count == 0:
        precision_total = 0
    else:
        precision_total = total_TP_count / (total_TP_count + total_FP_count)
    if total_TP_count + total_FN_count == 0:
        recall_total = 0
    else:
        recall_total = total_TP_count / (total_TP_count + total_FN_count)
    if precision_total + recall_total == 0:
        f1_score_total = 0
    else:
        f1_score_total = 2 * (precision_total * recall_total) / (precision_total + recall_total)

    # Organize all results in a dictionary
    results = {'seen': {'accuracy': accuracy_seen, 'precision': precision_seen, 'recall': recall_seen,
                        'f1': f1_score_seen, 'cf_matrix': cf_matrix_seen},
               'unseen': {'accuracy': accuracy_unseen, 'precision': precision_unseen, 'recall': recall_unseen,
                          'f1': f1_score_unseen, 'cf_matrix': cf_matrix_unseen},
               'total': {'accuracy': accuracy_total, 'precision': precision_total, 'recall': recall_total,
                         'f1': f1_score_total, 'cf_matrix': cf_matrix_total},
               'sklearn_results': {'accuracy': sklearn_results['accuracy'], 'precision': sklearn_results['precision'],
                                   'recall': sklearn_results['recall'], 'f1': sklearn_results['f1'],
                                   'cf_matrix': sklearn_results['cf_matrix'],
                                   'classification_report': sklearn_results['classification_report']}
               }
    return results


def binary_eval(predictions, references):
    """
    Evaluate the model on the given data.

    :param labels: The true labels of the data.
    :param predictions: The predictions of the model.
    :return: The accuracy, precision, recall, and f1-score of the model.
    """
    TN, FP, FN, TP = 0, 0, 0, 0
    for data_index, (labels_item, predictions_item) in enumerate(zip(references, predictions)):
        if len(labels_item) != len(predictions_item):
            print("ERROR: Length of labels and predictions do not match.")
            return
        for word_index, (label, prediction) in enumerate(zip(labels_item, predictions_item)):
            # Change 'O' to 0 if needed
            if label == 'O':
                label = 0
            if prediction == 'O':
                prediction = 0
            if label != 0 and prediction == 0:
                FN += 1
            elif label == 0 and prediction != 0:
                FP += 1
            elif label == 0 and prediction == 0:
                TN += 1
            elif label != 0 and prediction != 0:
                TP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}


def predictions_to_json(data, predictions, filename):
    """
    :param data: original string
    :param predictions: one hot vector with predictions. 1 for METAPHOR, 0 for non-METAPHOR
    :param filename: name of the file to save the json to
    :return: Json file with "text" as the data and "spans" as a dictionary with "start" and "end" as the "label"
    """
    json_data = []
    for data_index, (data_item, predictions_item) in enumerate(zip(data, predictions)):
        if len(data_item) != len(predictions_item):
            print("ERROR: Length of data and predictions do not match.")
            continue
        curr_index = 0
        spans = []
        full_text = ""
        for word_index, (word, prediction) in enumerate(zip(data_item, predictions_item)):
            end_index = curr_index + len(word)
            if prediction != 0:
                spans.append({"start": curr_index, "end": end_index, "label": "METAPHOR"})
            curr_index = end_index + 1  # +1 for the space
            # Add a space between words
            full_text += word + " "
        # remove last space
        full_text = full_text[:-1]
        json_data.append({"text": full_text, "spans": spans})
    with jsonlines.open(filename, 'w') as writer:
        for json_item in json_data:
            writer.write(json_item)


def create_confusion_matrix(cf_matrix, title, filename):
    """
    :param cf_matrix: confusion matrix
    :param cf_matrix:
    :return:
    """
    ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title(title)
    ax.set_xlabel('\nActual Values')
    ax.set_ylabel('Predicted Values')

    ax.xaxis.set_ticklabels(['Metaphor', 'Not Metaphor'])
    ax.yaxis.set_ticklabels(['Metaphor', 'Not Metaphor'])

    # tight layout fixes the figure size
    plt.tight_layout()
    plt.savefig('{}_{}.png'.format(filename, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    plt.show()


def get_metaphor_words(full_sentence, original_labels):
    full_sentence_words = full_sentence.split()
    metaphor_words = []
    if len(full_sentence_words) != len(original_labels):
        print("ERROR: Length of sentence and labels do not match.")
        # raise Exception("ERROR: Length of sentence and labels do not match.")
    for word, labels in zip(full_sentence_words, original_labels):
        if labels != 0:
            metaphor_words.append(word)
    return ",".join(metaphor_words)


def get_word(input_ids, w_index, tokenizer, word):
    """
    :param input_ids: input_ids of the sentence
    :param w_index: index of the word
    :param tokenizer: tokenizer used to tokenize the sentence
    :return: word at the given index
    """
    full_text = tokenizer.decode(input_ids)
    text_words = full_text.split()
    if word != text_words[w_index]:
        print("ERROR: Word at index {} is not the same as the given word.".format(w_index))
    if w_index >= len(text_words):
        print("ERROR: Index out of range")
        return ""
    return text_words[w_index]


def prepare_interlaced_labels(full_sentence, original_labels, tokenizer):
    full_sentence_split = full_sentence.split()
    prepared_labels = [word + ' ' + label_names[int(label)] for (label, word) in zip(original_labels, full_sentence_split)
                       if label != -100]
    return prepared_labels


def prepare_for_mt5(original_labels, input_ids, aligned_labels, method, tokenizer, max_target_length,
                    w_index=0, word=""):
    data = None
    full_sentence = tokenizer.decode(input_ids)
    full_sentence = full_sentence.replace('</s>', '')
    if method == 'interlaced':
        labels = prepare_interlaced_labels(full_sentence, original_labels, tokenizer)
    elif method == 'tag':
        labels = [label_names[label] for label in original_labels if label != -100]
    elif method == 'natural':
        data = [tokenizer.decode(input_ids) +'. המילים הבאות הן מטאפורות: <extra_id_0>']
        labels = ["<extra_id_0>:" + get_metaphor_words(full_sentence, original_labels)]
    elif method == 'natural_per_word':
        data = [tokenizer.decode(input_ids) +'.השימוש במילה {} במשפט הקודם הוא במשמעות <extra_id_0>.'.format(get_word(input_ids,
                                                                                                  w_index, tokenizer, word))]
        labels = ["<extra_id_0>:{}".format('מטאפורית' if aligned_labels == 1 else 'מדויקת')]
    elif method == 'zero_one':
        labels = [label for label in original_labels if label != -100]
    else:
        raise ValueError("Method not supported")
    return data, labels


def tokenize_and_align_labels_mt5(dataset, tokenizer, method='tag',
                                  ignore_subtokens=False, max_input_length=512, max_target_length=512):
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
    # print("???????????????????? tokenizer: ", tokenizer)
    # print("!!!!!! dataset'data'][0] = ", dataset["data"][0])
    # print("!!!!!! tokenized_inputs['input_ids'][0] = ", tokenized_inputs['input_ids'][0])

    all_labels = dataset["labels"]
    updated_labels_array = []
    updated_data_array = []
    w_index = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        input_ids = tokenized_inputs["input_ids"][i]
        if method == 'natural_per_word':
            alighned_labels = labels
            w_index = dataset['w_index']
            word = dataset["word"]
            new_data, new_label = prepare_for_mt5(labels, input_ids, alighned_labels, method, tokenizer,
                                                  max_target_length, w_index[i], word[i])
        else:
            alighned_labels = align_labels_with_tokens(labels, word_ids)
            new_data, new_label = prepare_for_mt5(labels, input_ids, alighned_labels, method, tokenizer,
                                                  max_target_length)

        updated_labels_array.append(new_label)
        if new_data is not None:
            updated_data_array.append(new_data)

    with tokenizer.as_target_tokenizer():
        updated_labels = tokenizer(updated_labels_array, max_length=max_target_length, truncation=True, is_split_into_words=True)

    if len(updated_data_array) > 0:
        tokenized_inputs = tokenizer(
            updated_data_array, truncation=True, is_split_into_words=True
        )

    tokenized_inputs["labels"] = updated_labels['input_ids']
    # tokenized_inputs["word_ids"] = tokenized_inputs.word_ids
    return tokenized_inputs


def get_model_checkpoint(model_type, curr_model_checkpoint):
    model_checkpoint = curr_model_checkpoint
    if 'alephbertgimmel' not in curr_model_checkpoint and \
            '/home/tok/figurative-language' not in curr_model_checkpoint:
        # switch case depending on model type
        if model_type == 'rebi_bert':
            model_checkpoint = 'rebi_bert'
        elif model_type == 'aleph_bert':
            model_checkpoint = 'onlplab/alephbert-base'
        elif model_type == 'mT5' or model_type == 'mt5':
            optional_model_checkpoints = ['google/mt5-small', 'google/mt5-base', 'google/mt5-large', 'google/mt5-xl',
                                          'google/mt5-xxl']
            if curr_model_checkpoint not in optional_model_checkpoints:
                model_checkpoint = 'google/mt5-small'
        else:
            raise ValueError('model type not supported')
    return model_checkpoint


def get_dataset_name(model_type, curr_dataset):
    if model_type == 'aleph_bert' or model_type == 'rebi_bert':
        dataset = curr_dataset
        dataset_name = dataset.split("/")[-1].split("Dataset.py")[0]
    elif model_type == 'simple_melbert':
        dataset = 'datasets/MetaphorRowsDataset.py'
        dataset_name = 'MetaphorRowsDataset'
    elif model_type == 'per_word':
        split_type = 'Rows'  # 'Para'
        dataset = 'datasets/PerWord{}Dataset.py'.format(split_type)
        dataset_name = 'PerWord{}Dataset'.format(split_type)
    elif model_type == 'mT5Encoder':
        dataset = 'datasets/MetaphorRowsDataset.py'
        dataset_name = 'MetaphorRowsDataset'
    elif model_type == 'mT5':
        dataset = 'datasets/MetaphorRowsDataset.py'
        dataset_name = 'MetaphorRowsDataset'
    else:
        dataset = 'datasets/MetaphorRowsDataset.py'
        dataset_name = 'MetaphorRowsDataset'
    return dataset, dataset_name


def get_experiment_name(model_type, model_checkpoint, random_weights, weighted_loss, non_metaphor_weight,
                        metaphor_weight, per_device_train_batch_size, gradient_accumulation_steps,
                        epochs, lr, layers, use_more_layers, dataset_method, only_intermediate_representation,
                        warmup_steps, seed, corpus):
    # if 'pre' in checkpoint set after_pretraining to dapt_
    after_pretraining = "dapt_" if 'after_pretraining' in model_checkpoint else ""
    if after_pretraining == "dapt_":
        order = "unordered_" if 'unordered' in model_checkpoint else "ordered_"
    else:
        order = ""

    if model_checkpoint == '/home/tok/figurative-language/xlm-roberta-base-finetuned_fbt_epochs_from_unsupervised':
        # iterate over folders in model_checkpoint
        for folder in os.listdir(model_checkpoint):
            if folder.startswith('checkpoint'):
                model_checkpoint = os.path.join(model_checkpoint, folder)
                break
    model_name = 'xlm_' if 'xlm' in model_checkpoint else 'bert_'
    if weighted_loss:
        weighted_loss_args = '{},{}'.format(non_metaphor_weight,
                                               metaphor_weight)
    else:
        weighted_loss_args = '_unweighted'

    version = model_checkpoint.split('-')[-1]

    experiment_name = "{},{}{}{}{}{},bs-{},gas-{},ep-{},wl-{},lr-{}{}{}{}{}{}{}".format(
        model_type,
        model_name,
        ",v-{}".format(version) if version != "" else "",
        ",rand" if random_weights else "",
        ",apt" if after_pretraining else "",
        ",nor" if not order else "",
        per_device_train_batch_size,
        gradient_accumulation_steps,
        epochs,
        weighted_loss_args,
        lr,
        ",ml-{}".format(layers) if use_more_layers else '',
        ',oip' if only_intermediate_representation else "",
        ",dm-{}".format(dataset_method) if model_type == 'mT5' else "",
        ",ws-{}".format(warmup_steps) if warmup_steps > 0 else "",
        ",s-{}".format(seed),
        ",c-{}".format(corpus)
    )

    return experiment_name

def generate_word_statistics(dataset_method, model_type, experiment_name, raw_datasets):
    path_words_statistics = os.path.join('results', 'stats', experiment_name)

    if dataset_method != 'natural_per_word' and (model_type != 'per_word' and not os.path.exists(path_words_statistics)):
        os.makedirs(path_words_statistics)
        # Counts the number of times a word appears in the dataset as a metaphor and as non-metaphor
        metaphor_words, non_metaphor_words = count_words(raw_datasets["train"])
        # Save the counts to a pickle file with the experiment name
        with open(os.path.join(path_words_statistics, 'metaphor_words.pkl'), 'wb') as f:
            pickle.dump(metaphor_words, f)
        with open(os.path.join(path_words_statistics, 'non_metaphor_words.pkl'), 'wb') as f:
            pickle.dump(non_metaphor_words, f)
    return path_words_statistics


def load_tokenizer(model_checkpoint, model_type, rebi_bert_path, raw_datasets, ignore_subtokens, dataset_method):
    if 'xlm' in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    elif "avichr/heBERT" in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
    elif model_type == 'mT5':
        tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
    elif model_type == 'rebi_bert':
        tokenizer = RabbinicTokenizer(BertTokenizer.from_pretrained(os.path.join(rebi_bert_path, 'vocab.txt')))
        tokenizer_object = convert_slow_tokenizer.convert_slow_tokenizer(tokenizer)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')

    if model_type == 'per_word':
        tokenized_datasets = raw_datasets.map(
            tokenize_sentence_word,
            batched=True,
            batch_size=None,
            remove_columns=raw_datasets["train"].column_names,
            fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': ignore_subtokens},
            load_from_cache_file=False
        )
    elif model_type == 'mT5':
        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels_mt5,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            fn_kwargs={'tokenizer': tokenizer,
                       'ignore_subtokens': ignore_subtokens,
                       'method': dataset_method,
                       },
            writer_batch_size=300,
            load_from_cache_file=False
        )
    else:
        # Tokenizes and aligns the labels
        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': ignore_subtokens},
            load_from_cache_file=False
        )
    return tokenizer, tokenized_datasets


def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 0.00001, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
        # "seed": trial.suggest_int("seed", 41, 42),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
        "weight_decay": trial.suggest_float("weight_decay", 0.00001, 0.1, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.5),
    }


def get_model_specific_args(model_type):
    melbert = False
    if model_type == 'aleph_bert' or model_type == 'mT5_encoder' or model_type == 'rebi_bert':
        compute_metrics = compute_metrics_baseline
        per_example_label = False
    elif model_type == 'simple_melbert':
        compute_metrics = compute_metrics_melbert
        per_example_label = True
        melbert = True
    else:
        compute_metrics = compute_metrics_baseline
        per_example_label = True
    return melbert, compute_metrics, per_example_label


def save_predictions(predictions, predictions_dir, experiment_name, tokenized_datasets, tokenizer, raw_datasets):
    test_dataset = raw_datasets['test']
    tokenized_test_dataset = tokenized_datasets['test']
    predictions = get_actual_predictions(predictions, tokenized_test_dataset, tokenizer, test_dataset,
                                         per_word=False)

    def make_binary_predictions(predictions, per_word):
        """
        predictions is a list of non-binary predictions
        Makes predictions binary (0 or 1)
        """
        if per_word:
            predictions = [1 if p == 2 else p for p in predictions]
        else:
            for list_index, pred_list in enumerate(predictions):
                for pred_index, pred in enumerate(pred_list):
                    if pred == 2:
                        predictions[list_index][pred_index] = 1
        return predictions

    # Make binary predictions
    predictions = make_binary_predictions(predictions, per_word=False)



    # Create a dataframe with the data, labels and predictions

    results_df = pd.DataFrame(columns=['data', 'labels', 'predictions'])
    for i, sample in enumerate(test_dataset):
        if len(sample['data']) != len(predictions[i]):
            print("Error: sample.data and predictions[i] have different lengths")
            return
        results_df.loc[i] = [sample['data'], sample['labels'], predictions[i]]

    # if the results folder does not exist, create it
    results_path = os.path.join(predictions_dir, experiment_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save the predictions to a json file.
    results_df.to_json(os.path.join(results_path, "predictions.json"), orient='records')

    # Save the predictions to a csv file.
    results_df.to_csv(os.path.join(results_path, "predictions.csv"), index=False)

    # Save the predictions to a pickle file.
    pd.to_pickle(results_df,os.path.join(results_path,"predictions.pkl"))

    # results_df.to_csv('{}/predictions.csv'.format(results_path), index=False)
    print('Predictions saved in {}/predictions.json'.format(results_path))


def load_dataset_split(corpus, split):
    """
    Loads a dataset split
    """
    if split == 'test':
        folder = 'test'
    else:
        folder = 'train'
    if split == 'train':
        split = datasets.Split.TRAIN
    elif split == 'test':
        split = datasets.Split.TEST
    elif split == 'validation':
        split = datasets.Split.VALIDATION
    data_dir = training_args.paths.data_dir

    dataset = pd.read_json(os.path.join(data_dir, "{}".format(folder), '{}_3_labels_{}.json'.format(split, corpus)))

    # if corpus == 'all':
    #     pinchas = pd.read_json(os.path.join(data_dir, "{}".format(folder), '{}_3_labels_pinchas_1.json'.format(split)))
    #     pre_piyut = pd.read_json(os.path.join(data_dir, "{}".format(folder), '{}_3_labels_pre_piyut_1.json'.format(split)))
    #     print("Loading dataset - one row per sample")
    #     dataset = pd.concat([pinchas, pre_piyut])
    # else:
    #     dataset = pd.read_json(os.path.join(data_dir, "{}".format(folder), '{}_3_labels_{}.json'.format(split, corpus)))

    dataset.rename(columns={'sentence': 'data', 'label': 'labels'}, inplace=True)
    dataset = dataset[['data', 'labels']]
    dataset = datasets.Dataset.from_pandas(dataset, split=split)
    return dataset

