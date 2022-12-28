import wandb
from transformers import AutoModelForTokenClassification, BertTokenizerFast, TrainingArguments, EarlyStoppingCallback
from transformers import AutoTokenizer, BertConfig
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from helpers.metric import compute_metrics_baseline, compute_metrics_melbert
from helpers.utils import *
import pickle
from config.config_parser import *
from config.set_config import set_specific_config
from torch import nn
from transformers import Trainer
from models.alephBert_per_word import AlephBertPerWord
from models.alephMelBert import alephMelBert
from trainers.wce_trainer import WeightedLossTrainer


def train_classification():
    print("Starting training of classification model")
    # Parse arguments
    check_args(training_args)
    dataset = 'datasets/MetaphorRowsDataset.py'
    dataset_name = 'MetaphorRowsDataset'
    checkpoint = training_args.model_args.model_checkpoint

    # Load layers for classification
    layers = ""
    if training_args.model_args.use_more_layers:
        layers = training_args.model_args.layers
        if ',' in layers:
            layers = layers.split(',')
        if len(layers) > 0:
            layers_for_classification = [int(layer) for layer in layers]
        else:
            layers_for_classification = []
    else:
        layers_for_classification = []


    # if 'pre' in checkpoint set pre to True
    after_pretraining = "dapt_" if 'after_pretraining' in checkpoint else ""
    if after_pretraining == "dapt_":
        order = "unordered_" if 'unordered' in checkpoint else "ordered_"
    else:
        order = ""
    lr = training_args.model_args.learning_rate
    epochs = training_args.model_args.num_epochs
    non_metaphor_weight = training_args.model_args.non_metaphor_weight
    metaphor_b_weight = training_args.model_args.metaphor_b_weight
    metaphor_i_weight = training_args.model_args.metaphor_i_weight
    weighted_loss = training_args.model_args.weighted_loss
    model_checkpoint = training_args.model_args.model_checkpoint
    if model_checkpoint == '/home/tok/figurative-language/xlm-roberta-base-finetuned_fbt_epochs_from_unsupervised':
        # iterate over folders in model_checkpoint
        for folder in os.listdir(model_checkpoint):
            if folder.startswith('checkpoint'):
                model_checkpoint = os.path.join(model_checkpoint, folder)
                break
    model_name = 'xlm_' if 'xlm' in model_checkpoint else 'bert_'
    if weighted_loss:
        weighted_loss_args = '{},{},{}'.format(non_metaphor_weight,
                                               metaphor_b_weight,
                                               metaphor_i_weight)
    else:
        weighted_loss_args = '_unweighted'
    ignore_subtokens = training_args.data_args.ignore_subtokens
    version = model_checkpoint.split('-')[-1]
    per_device_train_batch_size = training_args.model_args.per_device_train_batch_size
    experiment_name = "{}_melbert{}{}{}bs_{}_ep_{}_wl_{}_lr_{}_v_{}_lc_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                               model_name, after_pretraining, order, per_device_train_batch_size, epochs,
                                                           weighted_loss_args, lr, version, layers)
    # Set seed for reproducibility
    set_seed(training_args.seed)

    wandb_config = {
        "model_type": model_name,
        "model_name_or_path": model_checkpoint,
        "per_device_train_batch_size": per_device_train_batch_size,
        "num_epochs": epochs,
        "learning_rate": lr,
        "weighted_loss": weighted_loss,
        "non_metaphor_weight": non_metaphor_weight,
        "metaphor_b_weight": metaphor_b_weight,
        "metaphor_i_weight": metaphor_i_weight,
        "model_checkpoint": model_checkpoint,
        "ignore_subtokens": ignore_subtokens,
        "layer_index_for_classification": layers[0] if len(layers) > 0 else "",
        "dataset_name": dataset_name, "experiment_name": experiment_name
    }

    # initialize wandb to visualize training progress
    wandb.init(project="fl_finetuning", entity="tokeron", name=experiment_name, config=wandb_config)

    # Load dataset
    raw_datasets = load_dataset(dataset)

    path_words_statistics = os.path.join('results', 'stats', experiment_name)
    set_specific_config("config/config.yaml", "paths", "path_words_statistics", path_words_statistics)

    # Label names for the dataset labels
    label_names = ["O", "B-metaphor", "I-metaphor"]
    # Define dictionaries fot the model
    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    if training_args.model_args.use_more_layers:
        output_hidden_states = True
    else:
        output_hidden_states = False

    # Change the classification head length according to the additional layers list
    additional_layers_num = len(layers_for_classification)
    only_intermid_rep = training_args.model_args.only_intermediate_representation

    # Load a model with custom number of layers in the classification head
    model = alephMelBert(layers_for_cls=layers_for_classification, only_intermid_rep=only_intermid_rep)

    random_weights = training_args.model_args.random_weights
    # initialize model weights with random weights
    if random_weights:
        model.init_weights()

    # Load tokenizer
    if 'xlm' in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    elif "avichr/heBERT" in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
    else:
        tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')


    # Tokenizes and aligns the labels
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': training_args.data_args.ignore_subtokens},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Initialize training arguments for trainer
    args = TrainingArguments(
        "Aleph-bert-more-pretraining-metaphor",
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=training_args.model_args.logging_steps,
        eval_steps=training_args.model_args.eval_steps,
        save_strategy="steps",
        learning_rate=training_args.model_args.learning_rate,
        num_train_epochs=training_args.model_args.num_epochs,
        weight_decay=training_args.model_args.weight_decay,
        run_name=experiment_name,
        report_to="wandb",
        do_train=training_args.model_args.do_train,
        do_eval=training_args.model_args.do_eval,
        do_predict=training_args.model_args.do_predict,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * 2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    if weighted_loss:
        trainer = WeightedLossTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_melbert,
            tokenizer=tokenizer,
            layers_for_classification=layers_for_classification,
            per_example_label=True,
            melbert=True,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.model_args.early_stopping_patience)],
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_melbert,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.model_args.early_stopping_patience)],
        )


    print("Start training...")
    trainer.train()
    print("Finished training")


    # Save the model
    checkpoints_dir = training_args.paths.results_checkpoints_dir
    # create a directory for each experiment type - bert, bert_pretrained, xlm, xlm_pretrained
    experiment_type_dir = '{}_{}'.format(model_name, after_pretraining)
    checkpoints_dir = os.path.join(experiment_type_dir, checkpoints_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    trainer.save_model(checkpoints_dir + "/" + experiment_name)
    print("Saved model to {}".format(checkpoints_dir + "/" + experiment_name))
    # save model with torch.save
    torch.save(model.state_dict(), checkpoints_dir + "/" + experiment_name + '_torch')
    print("Saved model with torch to {}".format(checkpoints_dir + "/" + experiment_name))
    print("Done")
    # load model with torch.load
    # model.load_state_dict(torch.load(checkpoints_dir + "/" + experiment_name))


if __name__ == '__main__':
    train_classification()
