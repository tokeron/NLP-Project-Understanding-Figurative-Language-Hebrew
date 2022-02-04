import wandb
from transformers import AdamW, AutoModelForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from metric import compute_metrics
from DataExploration import print_predictions
import argparse
from box import Box
import yaml
from utils import check_args, set_seed, get_actual_predictions, tokenize_and_align_labels, save_heatmap
from DataExploration import explore_data
import pandas as pd


if __name__ == '__main__':
    #  Explore the data
    # explore_data()
    print("Starting run.py")
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train AlephBERT to find metaphors in Hebrew.')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    check_args(training_args)
    split_options = ["rows", "paragraphs"]
    split_by = split_options[training_args.data_args.split_by]
    lr = training_args.model_args.learning_rate
    epochs = training_args.model_args.num_epochs
    ignore_subtokens = training_args.data_args.ignore_subtokens
    experiment_name = "split-"+split_by + "_lr-" + str(lr) + "_epochs-" \
                      + str(epochs) + "_ignore-" + str(ignore_subtokens)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # initialize wandb to visualize training progress
    wandb.init(project="fl", entity="tokeron", name=experiment_name)

    # Load dataset
    raw_datasets = load_dataset(training_args.data_args.dataset)

    # Label names for the dataset labels
    label_names = ["O", "B-metaphor", "I-metaphor"]
    # Define dictionaries fot the model
    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    # Load pretrained AlephBERT model with a token classification head
    model = AutoModelForTokenClassification.from_pretrained(
        training_args.model_args.model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(training_args.model_args.model_checkpoint)

    # Tokenizes and aligns the labels
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': training_args.data_args.ignore_subtokens},
    )

    # Putting together samples inside a batch
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Initialize training arguments for trainer
    args = TrainingArguments(
        "Aleph-bert-finetuned-metaphor",
        evaluation_strategy="epoch",
        # save_strategy="epoch",
        learning_rate=training_args.model_args.learning_rate,
        num_train_epochs=training_args.model_args.num_epochs,
        weight_decay=training_args.model_args.weight_decay,
        run_name=experiment_name,
        report_to="wandb",
        do_train=training_args.model_args.do_train,
        do_eval=training_args.model_args.do_eval,
        do_predict=training_args.model_args.do_predict,

    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # trainer.train()
    # print("Finished training")
    #
    # # Evaluate the model on the validation set
    # predictions = trainer.predict(tokenized_datasets["validation"])
    # predictions = get_actual_predictions(predictions, tokenized_datasets['validation'], tokenizer)
    # print_predictions(raw_datasets['validation']['data'], raw_datasets['validation']['labels'], predictions,
    #                   filename="validation_predictions.docx")
    #
    # Evaluate the model on the test set
    predictions = trainer.predict(tokenized_datasets["test"])
    predictions = get_actual_predictions(predictions, tokenized_datasets['test'], tokenizer)
    print_predictions(raw_datasets['test']['data'], raw_datasets['test']['labels'], predictions,
                      filename="test_predictions.docx")
    #  Explore the results
    save_heatmap(raw_datasets['test']['data'], raw_datasets['test']['labels'], predictions,
                    filename="test_predictions")

    # Evaluate the model on the train set
    predictions = trainer.predict(tokenized_datasets['train'])
    predictions = get_actual_predictions(predictions, tokenized_datasets['train'], tokenizer)
    print_predictions(raw_datasets['train']['data'][:100], raw_datasets['train']['labels'][:100], predictions[:100],
                      filename="train_predictions" + experiment_name)

    # Save the model
    trainer.save_model(training_args.model_args.output_dir)
    print("Saved model to {}".format(training_args.model_args.output_dir))
