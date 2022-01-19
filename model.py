from sklearn.metrics import f1_score
from transformers import AdamW, AutoModelForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch.nn.functional as F
import pandas as pd
import torch
from datasets import load_dataset, load_metric
import FLDataset
from transformers import DataCollatorForTokenClassification
import numpy as np
import tokenizerHelper


if __name__ == "__main__":
    # Define model name - alephbert
    model_checkpoint = 'onlplab/alephbert-base'

    # Load dataset
    # ToDo: create a dataset class that contains train and test data.
    #  This way we can use the same code for both train and test

    train = pd.read_json('data/train_rows.json')
    test = pd.read_json('data/test_rows.json')
    validation = pd.read_json('data/validation_rows.json')

    train_dataset = FLDataset.FLDataset(train)

    # raw_datasets = load_dataset("json", data_files="data/train_rows.json", field="data")
    # raw_datasets = load_dataset("hebrew-metaphors-dataset")

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

    # Tokenizes and aligns the labels
    tokenized_datasets = raw_datasets.map(
        tokenizerHelper.tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # The function that responsible for putting together samples inside a batch
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Metric to evaluate the model
    metric = load_metric("seqeval")



# # Change the name of the classification predictions
# alephbert_token_classification.config.id2label = {0: "FL", 1: "NFL"}
#
# # Use the model tokenizer to create the input for the model
# max_seq_length = 512
# num_proc = 4
#
#
# def tokenize_and_align_labels(df):
#     df.insert(2, "tokenized", "")
#     for index, text in df.iterrows():
#         fulltext = df.at[index, "fulltext"]
#         tokenized_texts = alephbert_tokenizer(fulltext, padding=True, truncation=True, max_length=512,
#                                               return_tensors='pt', return_length=True)
#         labels = []
#         word_ids = tokenized_texts.word_ids()
#         print(fulltext)
#         print(index)
#         print("len:", len(df.at[index, 'FL']))
#         print("word_ids max: ", word_ids[-2])
#         for word_id in word_ids:
#             if word_id is None:
#                 labels.append(0)
#             else:
#                 print(word_id)
#                 labels.append(df.at[index, 'FL'][word_id])
#         df.at[index, 'tokenized'] = tokenized_texts.input_ids
#         df.at[index, 'FL'] = labels
#
#
#
#
# tokenize_and_align_labels(train_df)
# # tokenize_and_align_labels(test_df)
# # tokenize_and_align_labels(validation_df)
#
# train = FLDataset.FLDataset(train_df)
# test = FLDataset.FLDataset(test_df)
# validation = FLDataset.FLDataset(validation_df)
#
# train.set_format('torch')

# Prepare labels
# for split in batch:
    # batch[split] = batch[split].add_column('label', raw_datasets[split]['label'])

# args = TrainingArguments(output_dir=out_path, overwrite_output_dir=True, per_device_train_batch_size=16,
#                          per_device_eval_batch_size=16, metric_for_best_model='dev_f1',
#                          greater_is_better=True, do_train=True,
#                          num_train_epochs=8, evaluation_strategy="epoch")

# trainer = Trainer(
#     model=alephbert_token_classification,
#     args=args,
#     train_dataset=batch['train'],
#     eval_dataset=batch['validation'],
#     compute_metrics=metric_fn)

# # Training The Model
# trainer.train()
#
# print(trainer.evaluate())



# # Train the alephbert_token_classification network (fine-tuning)
# optimizer = AdamW(alephbert_token_classification.parameters())
# loss = alephbert_token_classification(**batch).loss
# loss.backward()
# optimizer.step()
#
# # disable dropout before evaluating
# alephbert_token_classification.eval()
#
# # Make predictions
# outputs = alephbert_token_classification(**batch)
# predictions = F.softmax(outputs.logits, dim=-1)
# print(alephbert_token_classification.config.id2label)
# print(torch.argmax(predictions, dim=-1))
#
# # Save the pretrained model
# alephbert_token_classification.save_pretrained("alephbert-fl-model")