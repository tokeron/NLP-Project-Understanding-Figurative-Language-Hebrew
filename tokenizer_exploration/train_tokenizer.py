from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizerFast
from explore_tokenizer import explore_tokenizer
import wandb
import pandas as pd
import os


def train_tokenizer():
    num_examples = 100

    old_tokenizer = BertTokenizerFast.from_pretrained('byblic_tokenizer')

    # Print example of tokenizer before training
    print("Example of tokenizer before training:")
    old_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    print('\n\n\n')
    df_res_before = explore_tokenizer(old_tokenizer, num_examples=num_examples)

    # Initialize wandb
    wandb.init(project="train_tokenizer", entity="tokeron", name="tokenizer_exploration")

    tokenizer_path = '/home/tok/figurative-language/tokenizer_exploration/byblic_tokenizer'
    if os.path.exists(tokenizer_path):
        new_tokenizer = BertTokenizerFast.from_pretrained('byblic_tokenizer')
    else:
        genres_list = ['tanach',
                       'mishna',
                       'tosefta',
                       'yerushalmi',
                       'midrashtanchuma',
                       'midrashtehilim',
                       'midrashraba',
                       'search',
                       'dicts',
                       'papers',
                       'letters',
                       'momories',
                       'proza',
                       'poems',
                       'meshalim',
                       'songs',
                       'fl_unsupervised'
                       ]

        mlm_data_path = '/home/tok/figurative-language/data/mlm-data'
        full_path = ['{}/{}.csv'.format(mlm_data_path, corpora) for corpora in genres_list]

        unsupervised_dataset = load_dataset('csv', data_files=full_path)

        def get_training_corpus():
            return (
                unsupervised_dataset["train"][i: i + 1000]["text"]
                for i in range(0, len(unsupervised_dataset["train"]), 1000)
            )

        training_corpus = get_training_corpus()
        vocab_size = 52000
        new_tokenizer = old_tokenizer.train_new_from_iterator(text_iterator=training_corpus, vocab_size=vocab_size)
        new_tokenizer.save_pretrained("byblic_tokenizer")
        print("Tokenizer trained and saved to disk")

    # Print example of tokenizer after training
    print("Example of tokenizer after training:")
    df_res_after = explore_tokenizer(new_tokenizer, num_examples=num_examples)

    # add a column 'tokenized_sentence_after' to df_res_before
    df_res_before['tokenized_sentence_after_training'] = df_res_after['tokenized_sentence']

    # Upload dataframes to wandb
    wandb.log({"Tokenizer": df_res_before})


if __name__ == "__main__":
    train_tokenizer()
