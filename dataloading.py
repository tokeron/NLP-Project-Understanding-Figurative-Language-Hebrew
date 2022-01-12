import numpy as np
import pandas as pd
import sqlalchemy as db
from datasets import load_dataset_builder
import torch


def load_from_db():
    path_to_db = 'D:\\Documents\\A - studies\\A - Technion\\' \
             'semester 9\\Project - FL NLP\\NLP-Project\\Understanding-Figurative-Language-Hebrew\\data\\catmaviz.db'

    engine = db.create_engine('sqlite:///'+path_to_db)

    table_names = engine.table_names()
    pd_tables = []
    for table in table_names:
        pd_tables.append(pd.read_sql_table(table, engine))


def load_from_csv():
    annotation = pd.read_csv("data/annotations.csv")
    texts = pd.read_csv("data/texts.csv")
    return annotation, texts


def get_texts():
    texts = pd.read_csv("data/texts.csv")
    data = texts.fulltext
    return data.values.tolist()


def get_data():
    texts = pd.read_csv("data/texts.csv")
    annotations = pd.read_csv("data/annotations.csv")
    return annotations, texts


def count_spaces(fulltext, start_offset):
    text = fulltext[:start_offset]
    words = text.split()
    return len(words)


def generate_data():
    annotations, texts = get_data()
    fl_id_list = ['8ab01bf2-a50d-4ed3-8307-b00d5deddcfe']
    fl_annotations = annotations[(annotations.tagset_id.isin(fl_id_list))]
    texts = texts.drop(columns=['user_id', 'name', 'corpus_id', 'genre_id'])
    annotations.drop(columns=['id', 'user_id', 'path', 'phrase', 'tagset_id', 'tag_id'])
    texts.insert(3, "FL", "")
    texts.FL = texts.FL.astype(object)
    for index, text in texts.iterrows():
        fulltext = texts.at[index, "fulltext"]
        words_num = count_spaces(fulltext, len(fulltext) - 1)
        texts.at[index, "FL"] = np.zeros(words_num)
    for index, annotation in fl_annotations.iterrows():
        text_id = annotation['text_id']
        start_offset = annotation['start_offset']
        end_offset = annotation['end_offset']
        fulltext = texts[texts.id == text_id].fulltext
        word_number = count_spaces(fulltext.to_string(), start_offset)
        texts.loc[texts.id == text_id, "FL"].to_numpy()[0][word_number] = 1
    return texts


if __name__ == "__main__":
    texts = generate_data()
    texts.to_csv("data/prepared_data.csv")



