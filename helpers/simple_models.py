import os
import pickle
import pandas as pd
import datasets

def load_dataset_split_simple(corpus, split):
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
    data_dir = "/home/tok/figurative-language/figurative-language-data/prepared_data/"

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

def calculate_words_statistics_simple(raw_dataset, dataset_name, per_word=False):
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

def constant_false(dataset):
    predictions = []
    for i, example in enumerate(dataset):
        curr_pred = []
        for word, _ in zip(example['data'], example['labels']):
            curr_pred.append(0)
        predictions.append(curr_pred)
    return predictions


def majority(dataset, words_statistics):
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


def simple_models(corpus, results_dir):
    raw_datasets = datasets.DatasetDict({datasets.Split.TRAIN: load_dataset_split_simple(corpus, "train"),
                                        datasets.Split.VALIDATION: load_dataset_split_simple(corpus, "validation"),
                                        datasets.Split.TEST: load_dataset_split_simple(corpus, "test")})

    train = raw_datasets['train']
    val = raw_datasets['validation']
    test = raw_datasets['test']

    # Get the words statistics
    words_statistics, metaphor_words, non_metaphor_words = calculate_words_statistics_simple(train, dataset_name)

    # Get the predictions
    constant_false_predictions = constant_false(test)

    # Save the predictions
    predictions_dir = '/home/tok/figurative-language/figurative-language-data/results_vanilla_models'
    experiment_name = 'constant_false'

    results_df = pd.DataFrame(columns=['data', 'labels', 'predictions'])
    for i, sample in enumerate(test):
        if len(sample['data']) != len(constant_false_predictions[i]):
            print("Error: sample.data and predictions[i] have different lengths")
            return
        results_df.loc[i] = [sample['data'], sample['labels'], constant_false_predictions[i]]

    # if the results folder does not exist, create it
    results_path = os.path.join(predictions_dir, experiment_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save the predictions to a json file.
    results_df.to_json(os.path.join(results_path, "predictions.json"), orient='records')

    # Save the predictions to a csv file.
    results_df.to_csv(os.path.join(results_path, "predictions.csv"), index=False)

    # Save the predictions to a pickle file.
    pd.to_pickle(results_df, os.path.join(results_path, "predictions.pkl"))

    # results_df.to_csv('{}/predictions.csv'.format(results_path), index=False)
    print('Predictions saved in {}/predictions.json'.format(results_path))

    # Save the predictions
    predictions_dir = '/home/tok/figurative-language/figurative-language-data/results_vanilla_models'
    experiment_name = 'majority'
    majority_predictions = majority(test, words_statistics)

    results_df = pd.DataFrame(columns=['data', 'labels', 'predictions'])
    for i, sample in enumerate(test):
        if len(sample['data']) != len(majority_predictions[i]):
            print("Error: sample.data and predictions[i] have different lengths")
            return
        results_df.loc[i] = [sample['data'], sample['labels'], majority_predictions[i]]

    # if the results folder does not exist, create it
    results_path = os.path.join(predictions_dir, experiment_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save the predictions to a json file.
    results_df.to_json(os.path.join(results_path, "predictions.json"), orient='records')

    # Save the predictions to a csv file.
    results_df.to_csv(os.path.join(results_path, "predictions.csv"), index=False)

    # Save the predictions to a pickle file.
    pd.to_pickle(results_df, os.path.join(results_path, "predictions.pkl"))

    # results_df.to_csv('{}/predictions.csv'.format(results_path), index=False)
    print('Predictions saved in {}/predictions.json'.format(results_path))


if __name__ == '__main__':
    prepared_data_path = 'prepared_data'
    results_dir = 'results_vanilla_models'
    dataset_name = 'pinchas_1' # 'pre_piyut_1' # 'pinchas_1'
    simple_models(dataset_name, results_dir)
