from sklearn.metrics import f1_score
from transformers import AdamW, AutoModelForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch.nn.functional as F
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from MetaphorDataset import MetaphorDataset
from transformers import DataCollatorForTokenClassification
import numpy as np
from metric import compute_metrics

#  Define model name - alephbert
model_checkpoint = 'onlplab/alephbert-base'

# Load dataset
raw_datasets = load_dataset('MetaphorDataset.py')

# Label names for the dataset labels
label_names = ["O", "B-metaphor", "I-metaphor"]

# Define dictionaries fot the model
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# Load pretrained AlephBERT model with a token classification head
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')


# Align the labels to the words after tokenization
def align_labels_with_tokens(labels, word_ids):
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
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


# Use align_labels_with_tokens on the entire dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["data"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# Tokenizes and aligns the labels
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# The function that responsible for putting together samples inside a batch
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

args = TrainingArguments(
    "Aleph-bert-finetuned-metaphor",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
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

trainer.train()
