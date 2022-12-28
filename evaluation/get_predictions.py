from transformers import AutoModelForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from transformers import AutoTokenizer, BertConfig, AutoModel, PreTrainedModel
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from helpers.metric import compute_metrics_baseline
import os
import pandas as pd
from config.config_parser import *
from helpers.utils import get_actual_predictions, tokenize_and_align_labels

from models.alephBert_per_word import AlephBertPerWord
from models.alephMelBert import alephMelBert
from trainers.wce_trainer import WeightedLossTrainer
from helpers.utils import *
from transformers import AutoModelForTokenClassification, BertTokenizerFast, TrainingArguments, EarlyStoppingCallback
from transformers import MT5Model, T5TokenizerFast, MT5EncoderModel
from models.alephBert_wide_head import AlephBertWideHead

from BEREL_PyTorch_Model.rabtokenizer import RabbinicTokenizer
from transformers import BertTokenizer, BertForMaskedLM
from transformers import convert_slow_tokenizer
from transformers import PreTrainedTokenizerFast

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MT5ForConditionalGeneration
from helpers.metric import *
from helpers.utils import *
from load_trainer import *
import box


def get_model_type(checkpoint_path):
    return checkpoint_path.split("/")[-1].split(",")[0]


def get_dataset_method(checkpoint_path):
    """
    Returns the dataset method used to train the model.
    """
    dm = ""
    if ',dm-' in checkpoint_path:
        dm = checkpoint_path.split(",dm-")[-1].split(",")[0]
    return dm


def is_intermediate_representation(checkpoint_path):
    """
    Returns True if the model is an intermediate representation model.
    """
    return ',oip' in checkpoint_path


def is_more_layers(checkpoint_path):
    """
    Returns True if the model is a more layers model.
    """
    return ',ml-' in checkpoint_path


def get_additional_layers(checkpoint_path):
    """
    Returns the number of additional layers used in the model.
    """
    layers = []
    if is_more_layers(checkpoint_path):
        str_layers = checkpoint_path.split(",ml-")[-1].split(",")[0]
        for str_layer in str_layers:
            str_layer = str_layer.replace("[", "")
            str_layer = str_layer.replace("]", "")
            str_layer = str_layer.replace(" ", "")
            layers.append(int(str_layer))
    return layers


def get_seed(checkpoint_path):
    """
    Returns the seed used to train the model.
    """
    seed = ""
    if ',s-' in checkpoint_path:
        seed = checkpoint_path.split(",s-")[-1].split(",")[0]
    else:
        seed = '42'
    return int(seed)


def get_predictions(checkpoint_path, results_path, test_dataset, dataset_split):
    """
    Evaluates the model on the test set and generate predictions.
    :param checkpoint_path: path to the checkpoint
    :param results_path: path to the results folder
    :return: predictions dataframe of the following format:
        - data: list of words
        - labels: list of zeros and ones (original labels - GT)
        - predictions: list of zeros and ones (predicted labels)
    """

    args = {
        'model_type': get_model_type(checkpoint_path),
        'checkpoint': checkpoint_path,
        'lr': 0,
        'bs': 64,
        'ep': 0,
        'w_decay': 0,
        'w_loss': False,
        'metaphor_w': 1.0,
        'nmetaphor_w': 1.0,
        'gas': 1,
        'random_w': False,
        'lm': get_dataset_method(checkpoint_path),
        'only_ir': is_intermediate_representation(checkpoint_path),
        'use_ml': is_more_layers(checkpoint_path),
        'add_l': get_additional_layers(checkpoint_path),
        # 'add_l_num': len(get_additional_layers(checkpoint_path)),
        'wandb_name': 'final_eval_fl',
        'seed': get_seed(checkpoint_path),
        'warmup_s': 0,
        'warmup_r': 0.0,
        'lr_sched': 'linear'
    }

    args = box.Box(args)

    experiment_name, trainer, tokenized_datasets, tokenizer = load_trainer_with_datasets(args, wandb_init=False)
    tokenized_dataset = tokenized_datasets[dataset_split]
    # Evaluate the model on data from another source
    predictions = trainer.predict(tokenized_dataset)
    # TODO: if use per_word, get per_word variable from the checkpoint name
    per_word = False
    predictions = get_actual_predictions(predictions, tokenized_dataset, tokenizer, test_dataset, per_word=per_word)

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
    predictions = make_binary_predictions(predictions, per_word=per_word)



    # Create a dataframe with the data, labels and predictions
    # iterate over test_dataset
    if per_word:
        results_df = pd.DataFrame(columns=['sentence', 'word', 'label', 'prediction'])
        for i, sample in enumerate(test_dataset):
            results_df.loc[i] = [sample['sentence'], sample['word'], sample['label'], predictions[i]]
    else:
        results_df = pd.DataFrame(columns=['data', 'labels', 'predictions'])
        for i, sample in enumerate(test_dataset):
            if len(sample['data']) != len(predictions[i]):
                print("Error: sample.data and predictions[i] have different lengths")
                return
            results_df.loc[i] = [sample['data'], sample['labels'], predictions[i]]

    # if the results folder does not exist, create it
    results_path = results_path + "_" + dataset_split
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save the predictions to a json file.
    results_df.to_json(results_path + "/predictions.json", orient='records')

    # Save the predictions to a csv file.
    results_df.to_csv(results_path + "/predictions.csv", index=False)

    # Save the predictions to a pickle file.
    pd.to_pickle(results_df, results_path + "/predictions.pkl")

    # results_df.to_csv('{}/predictions.csv'.format(results_path), index=False)
    print('Predictions saved in {}/predictions.json'.format(results_path))
    return results_df


