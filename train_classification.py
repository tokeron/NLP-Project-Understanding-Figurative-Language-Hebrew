import datetime
import wandb
from transformers import AutoModelForTokenClassification, BertTokenizerFast, TrainingArguments, EarlyStoppingCallback
from transformers import AutoTokenizer, BertConfig
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification, TrainerCallback
from helpers.metric import *
from helpers.utils import *
import pickle
from config.config_parser import *
# from config.set_config import set_specific_config
from torch import nn
from transformers import Trainer
from models.alephBert_per_word import AlephBertPerWord
from models.alephMelBert import alephMelBert
from trainers.wce_trainer import WeightedLossTrainer
from models.alephBert_wide_head import AlephBertWideHead
from evaluation.evaluate_my_models import evaluate_all_models
import copy
import transformers
from transformers import MT5Model, T5TokenizerFast, MT5EncoderModel
from BEREL_PyTorch_Model.rabtokenizer import RabbinicTokenizer
from transformers import BertTokenizer, BertForMaskedLM
from transformers import convert_slow_tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MT5ForConditionalGeneration
import argparse
from load_trainer import load_trainer_with_datasets

# define global variables
label_names = ["O", "B-metaphor", "I-metaphor"]
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


def f1_objective(metrics):
    return metrics['eval_f1']


# main function
def train_classification(args):
    # # for debugging
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    n_trials = args.n_trials
    hyperparameter_search = args.hyperparameter_search
    model_type = args.model_type
    corpus = args.corpus
    save_model = training_args.model_args.save_model
    save_torch_model = training_args.model_args.save_torch_model
    model_checkpoint = args.checkpoint
    save_strategy = training_args.model_args.save_strategy
    predictions_dir = training_args.paths.predictions_dir
    ignore_subtokens = training_args.model_args.ignore_subtokens

    print("Starting training of classification model")

    # Main function - loads datasets and trainers
    trainer_with_data = load_trainer_with_datasets(args)

    experiment_name = trainer_with_data["experiment_name"]
    trainer = trainer_with_data["trainer"]
    tokenized_datasets = trainer_with_data["tokenized_datasets"]
    tokenizer = trainer_with_data["tokenizer"]
    raw_datasets = trainer_with_data["raw_datasets"]

    if hyperparameter_search:
        best_trial = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space,
                                                    compute_objective=f1_objective, n_trials=n_trials)

        wandb.log({'Experiment': experiment_name,
                   'objective': best_trial.objective,
                   'hyperparameters': best_trial.hyperparameters,
                   'run_id': best_trial.run_id})

        table = wandb.Table(columns=["run_id", "objective", "hyperparameters"])
        table.add_data(best_trial.run_id, best_trial.objective, best_trial.hyperparameters)
        wandb.log({"Best Model": table})
        # Print the best trial hyperparameters and objective values into a txt file
        with open("best_trial.txt", "w") as f:
            f.write("Best trial hyperparameters: " + str(best_trial.hyperparameters) + "\n")
            f.write("Best trial objective: " + str(best_trial.objective) + "\n")
            f.write("Best trial run_id: " + str(best_trial.run_id) + "\n")

    else:
        print("Start training...")
        trainer.train()
        print("Finished training")

        # Save the model
        checkpoints_dir = training_args.paths.results_checkpoints_dir
        # create a directory for each experiment type
        after_pretraining = "dapt_" if 'after_pretraining' in model_checkpoint else ""
        experiment_type_dir = '{}_{}'.format(model_type, after_pretraining)
        checkpoints_dir = os.path.join(experiment_type_dir, checkpoints_dir)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if save_model:
            trainer.save_model(os.path.join(checkpoints_dir, experiment_name))
        # if save_torch_model:
        #     torch.save(model.s`tate_dict(), checkpoints_dir + "/" + experiment_name + "_torch_model")
        #     print("Saved model with torch to {}".format(checkpoints_dir + "/" + experiment_name))
        # evaluate the model on test set

        if corpus.endswith("1"):
            pinchasov_test = load_dataset_split("pinchas_1", "test")
            pinchasov_test = pinchasov_test.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=pinchasov_test.column_names,
                fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': ignore_subtokens},
                load_from_cache_file=False
            )

            pre_piyut_test = load_dataset_split("pre_piyut_1", "test")
            pre_piyut_test = pre_piyut_test.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=pre_piyut_test.column_names,
                fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': ignore_subtokens},
                load_from_cache_file=False
            )

            all_test = load_dataset_split("all_1", "test")
            all_test = all_test.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=all_test.column_names,
                fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': ignore_subtokens},
                load_from_cache_file=False
            )
            res_pinchasov = trainer.evaluate(pinchasov_test)
            res_pre_piyut = trainer.evaluate(pre_piyut_test)
            res_all = trainer.evaluate(all_test)
            wandb.log({'Experiment': experiment_name, 'pinchasov_f1': res_pinchasov['eval_f1'],
                       'pinchasov_precision': res_pinchasov['eval_precision'],
                       'pinchasov_recall': res_pinchasov['eval_recall'],
                       'pinchasov_accuracy': res_pinchasov['eval_accuracy']
                       })


            wandb.log({'Experiment': experiment_name, 'pre_piyut_f1': res_pre_piyut['eval_f1'],
                          'pre_piyut_precision': res_pre_piyut['eval_precision'],
                          'pre_piyut_recall': res_pre_piyut['eval_recall'],
                          'pre_piyut_accuracy': res_pre_piyut['eval_accuracy']
                          })
            wandb.log({'Experiment': experiment_name, 'all_f1': res_all['eval_f1'],
                            'all_precision': res_all['eval_precision'],
                            'all_recall': res_all['eval_recall'],
                            'all_accuracy': res_all['eval_accuracy']
                            })

        if save_model or save_strategy != "no":
            test_evaluation = trainer.evaluate(tokenized_datasets["test"])
            wandb.log({'Experiment': experiment_name, 'test_f1': test_evaluation['eval_f1'],
                          'test_precision': test_evaluation['eval_precision'],
                          'test_recall': test_evaluation['eval_recall'],
                          'test_accuracy': test_evaluation['eval_accuracy']
                          })

            val_evaluation = trainer.evaluate(tokenized_datasets["validation"])
            wandb.log({'Experiment': experiment_name, 'val_f1': val_evaluation['eval_f1'],
                            'val_precision': val_evaluation['eval_precision'],
                            'val_recall': val_evaluation['eval_recall'],
                            'val_accuracy': val_evaluation['eval_accuracy']
                            })

            train_evaluation = trainer.evaluate(tokenized_datasets["train"])
            wandb.log({'Experiment': experiment_name, 'train_f1': train_evaluation['eval_f1'],
                            'train_precision': train_evaluation['eval_precision'],
                            'train_recall': train_evaluation['eval_recall'],
                            'train_accuracy': train_evaluation['eval_accuracy']
                            })
            print("Finished evaluation on test set")
            print("predicting...")
            predictions = trainer.predict(tokenized_datasets["test"])
            print("Saving predictions...")
            save_predictions(predictions, predictions_dir, experiment_name, tokenized_datasets, tokenizer, raw_datasets)

            # results_path = checkpoints_dir
            # checkpoint_list = [checkpoints_dir + "/" + experiment_name]
            # evaluate_all_models(checkpoint_list, results_path, do_eval=True)
    print("Done everything")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classification model')

    # basic setup
    parser.add_argument('--wandb_name', type=str, default='fine_tuning_fl', help='Name of the experiment in wandb')
    parser.add_argument('--model_type', type=str, default='aleph_bert', help='Model type',
                        choices=['aleph_bert', 'rebi_bert', 'simple_melbert', 'per_word', 'mT5'])
    parser.add_argument('--checkpoint', type=str, default='bert-base-uncased', help='Model checkpoint')
    parser.add_argument('--lr', type=float, default=0.000054, help='Learning rate')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--ep', type=int, default=10, help='Number of epochs')
    parser.add_argument('--w_decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--gas', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--random_w', type=bool, default=False, help='Random weights')
    parser.add_argument('--warmup_s', type=int, default=100, help='Warmup steps')  # 400
    parser.add_argument('--warmup_r', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--lr_sched', type=str, default='linear', help='Learning rate scheduler type',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                 'constant_with_warmup'])
    # Weghted loss
    parser.add_argument('--w_loss', type=bool, default=True, help='Use weighted loss')
    parser.add_argument('--metaphor_w', type=float, default=1.0, help='Weight of metaphor loss') # 9.0
    parser.add_argument('--nmetaphor_w', type=float, default=1.0, help='Weight of non-metaphor loss')

    # hyperparameter search
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials for hyperparameter search')
    parser.add_argument('--hyperparameter_search', type=bool, default=False, help='Hyperparameter search')

    # mT5
    parser.add_argument('--lm', type=str, default='interlaced', help='Labeling method',
                        choices=['interlaced', 'tag', 'natural'])

    # intermediate representation
    parser.add_argument('--only_ir', type=bool, default=False, help='Only use intermediate representation')
    parser.add_argument('--use_ml', type=bool, default=False, help='Use more layers')
    parser.add_argument('--add_l', type=str, default='3_4', help='Additional layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--corpus', type=str, default='pre_piyut_1', help='Corpus',
                        choices=['pre_piyut_20', 'pre_piyut_1', 'pinchas_1', 'all_1'])
    parser.add_argument('--esp', type=int, default=3, help='Early stopping patience')

    args = parser.parse_args()

    train_classification(args)
