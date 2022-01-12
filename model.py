from transformers import AdamW, AutoModelForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch.nn.functional as F
import dataloading
import pandas as pd
import torch

# Load pretrained BERT model with a classification head
alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
alephbert_token_classification = AutoModelForTokenClassification.from_pretrained('onlplab/alephbert-base')

# Change the name of the classification predictions
alephbert_token_classification.config.id2label = {0: "FL", 1: "NFL"}

# Load the data and Use the model tokenizer to create the input for the model
raw_data = pd.read_csv("data/prepared_data.csv")
batch = alephbert_tokenizer(raw_data, padding=True, truncation=True, return_tensors="pt")
batch["labels"] = torch.zeros(batch.data['input_ids'].shape).type(torch.LongTensor)

# Split tha data
# train, test =

# Train the alephbert_token_classification network (fine-tuning)
optimizer = AdamW(alephbert_token_classification.parameters())
loss = alephbert_token_classification(**batch).loss
loss.backward()
optimizer.step()

# disable dropout before evaluating
alephbert_token_classification.eval()

# Make predictions
outputs = alephbert_token_classification(**batch)
predictions = F.softmax(outputs.logits, dim=-1)
print(alephbert_token_classification.config.id2label)
print(torch.argmax(predictions, dim=-1))

# Save the pretrained model
alephbert_token_classification.save_pretrained("alephbert-fl-model")