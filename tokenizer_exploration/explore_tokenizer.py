import pandas as pd
import wandb
from transformers import AutoModelForTokenClassification, BertTokenizerFast, TrainingArguments, EarlyStoppingCallback
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from helpers.metric import compute_metrics_baseline
from helpers.utils import *
import pickle
from torch import nn
from transformers import Trainer
import matplotlib.pyplot as plt
import numpy as np


def explore_tokenizer(tokenizer: AutoTokenizer, num_examples: int = 10):
    # Load the dataset
    raw_datasets = load_dataset("/home/tok/figurative-language/datasets/MetaphorRowsDataset.py")
    dataset = raw_datasets["train"]

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'ignore_subtokens': 'true'},
    )

    # explore the tokenizer results
    print(tokenized_datasets)

    df = pd.DataFrame(columns=["original sentence", "tokenized_sentence", "token 1", "token 2", "token 3",
                               "token 4", "token 5", "token 6"])
        # , "token 7", "token 8", "token 9"])
    np.random.seed(7)
    rand_idx = np.random.randint(0, len(dataset), num_examples)
    # Iterate over the dataset and tokenized dataset
    for i in rand_idx:
        if len(tokenized_datasets['train']['input_ids'][i]) > 7:
            continue
        data = dataset['data'][i]
        full_sentence = ''
        for word in data:
            full_sentence += word + ' '
        tokenized_sentence = ''
        for label, token in zip(tokenized_datasets['train']['labels'][i], tokenized_datasets['train']['input_ids'][i]):
            if label != -100:
                tokenized_sentence += tokenizer.decode(token) + ' '
        df = pd.concat([df, pd.DataFrame({"original sentence": [full_sentence], "tokenized_sentence": [tokenized_sentence],
                                          "token 1": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][1])]
                                          if len(tokenized_datasets['train']['input_ids'][i]) > 1 else [''],
                                          "token 2": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][2])]
                                          if len(tokenized_datasets['train']['input_ids'][i]) > 2 else [''],
                                          "token 3": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][3])]
                                          if len(tokenized_datasets['train']['input_ids'][i]) > 3 else [''],
                                          "token 4": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][4])]
                                          if len(tokenized_datasets['train']['input_ids'][i]) > 4 else [''],
                                          "token 5": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][5])]
                                          if len(tokenized_datasets['train']['input_ids'][i]) > 5 else [''],
                                          "token 6": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][6])]
                                          if len(tokenized_datasets['train']['input_ids'][i]) > 6 else ['']
                                          # "token 7": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][7])]
                                          # if len(tokenized_datasets['train']['input_ids'][i]) > 7 else [''],
                                          # "token 8": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][8])]
                                          # if len(tokenized_datasets['train']['input_ids'][i]) > 8 else [''],
                                          # "token 9": [tokenizer.decode(tokenized_datasets['train']['input_ids'][i][9])]
                                          # if len(tokenized_datasets['train']['input_ids'][i]) > 9 else ['']
                                            })])
    return df


if __name__ == '__main__':
    # Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    df = explore_tokenizer(tokenizer, num_examples=10)
    print(df)

    # drop tokenized_sentence column
    df = df.drop(columns=['tokenized_sentence'])

    # reverse the order of the columns : first is last and last is first
    df = df[df.columns[::-1]]


    # save df to a table with matplotlib
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    for example_i, example in enumerate(df.values):
        for split_i, split in enumerate(example):
            df.iloc[example_i, split_i] = get_display(split)

    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    for i in range (0, len(df.columns)):
        the_table.auto_set_column_width(i)

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    # the_table.scale(1, 1.1)

    # make table bigger
    # add a title
    ax.set_title('Tokenized examples - alephBERT tokenizer', fontsize=20)
    fig.tight_layout()

    plt.savefig('tokenizer_exploration{}.png'.format(random.randint(0, 1000)), dpi=300)
    plt.show()

