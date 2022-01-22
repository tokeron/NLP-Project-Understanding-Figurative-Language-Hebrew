import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Labels
label_names = {"O": 0, "B-metaphor": 1, "I-metaphor": 2}


# Load data from csv
def get_data():
    texts = pd.read_csv("data/texts_1.csv")
    annotations = pd.read_csv("data/annotations_1.csv")
    labels = pd.read_csv("data/tags_1.csv")
    return labels, annotations, texts


# Word length starting from start_offset in text
def check_word_len(fulltext, start_offset):
    text = fulltext[start_offset:]
    words = text.split()
    return len(words[0])


def generate_data():
    """
    # Function that generates the data for the model
    :return:
    Dataframe with columns: data: list of words, labels: list of labels
    """
    # Get data from csv
    labels, annotations, texts = get_data()
    bad_data_list = [52341, 52586] # , 52705, 52946, 53307, 53499
    annotations.drop(annotations[annotations.id.isin(bad_data_list)].index, inplace=True)
    # Get only relevant annotations
    fl_annotations = annotations.loc[(
                annotations.path.str.contains('כינוי ציורי', regex=False) |
                annotations.path.str.contains('מטפוריקה', regex=False))]

    # Add columns for labels and text split into words
    last_col_num = len(texts.columns)
    texts.insert(last_col_num, "labels", "")
    texts.insert(last_col_num + 1, "data", "")
    texts.labels = texts.labels.astype(object)
    texts.data = texts.data.astype(object)
    for index, text in texts.iterrows():
        fulltext = texts.at[index, "fulltext"]
        fulltext_split = fulltext.split()
        texts.at[index, "data"] = fulltext_split
        texts.at[index, "labels"] = np.zeros([len(fulltext_split)])

    # For each annotation, add the tag
    for index, annotation in fl_annotations.iterrows():
        text_id = annotation['text_id']
        fulltext = texts[texts.id == text_id].fulltext.to_numpy()[0]
        start_offset = annotation['start_offset']
        end_offset = annotation['end_offset']
        metaphor_phrase = annotation.phrase

        # Clean annotations from spaces at the end ( update the end_offset)
        space_list = ['\n', ' ', '\t', '\r']
        i = -1
        while len(metaphor_phrase) + i >= 0 and metaphor_phrase[i] in space_list:
            i = i - 1
            end_offset = end_offset - 1

        # Check if start_offset is in the middle of a word
        while start_offset > 0 and fulltext[start_offset - 1] not in space_list:
            start_offset = start_offset - 1

        # Check if end_offset is in the middle of a word
        while end_offset < len(fulltext) and fulltext[end_offset] not in space_list:
            end_offset = end_offset + 1

        remaining_metaphor_len = end_offset - start_offset
        word_index = len(fulltext[:start_offset].split())
        number_of_words_in_phrase = 0
        next_word_start_index = start_offset

        # Iterate over the words in the phrase (annotation) and add the labels
        while remaining_metaphor_len > 0:
            remaining_text = fulltext[next_word_start_index:]  # Get the remaining text
            word_len = len(remaining_text.split()[0])  # Get the length of the next word
            remaining_metaphor_len = remaining_metaphor_len - word_len - 1  # -1 for the space
            if number_of_words_in_phrase == 0:  # If it's the first word in the phrase
                texts.loc[texts.id == text_id, "labels"].to_numpy()[0][word_index] = label_names["B-metaphor"]
            else:  # If it's not the first word in the phrase
                texts.loc[texts.id == text_id, "labels"].to_numpy()[0][word_index] = label_names["I-metaphor"]
            word_index = word_index + 1
            next_word_start_index = next_word_start_index + word_len + 1  # +1 for the space
            number_of_words_in_phrase = number_of_words_in_phrase + 1  # +1 for the word
    return texts


def split_by_rows(texts, split_by="\r\n"):
    """
    # Function that splits the data by rows
    :param texts: Dataframe with columns: data: list of words, labels: list of labels
    :return:
    texts_rows: Dataframe with columns:
                    data: list of words (corresponding to rows in the original text),
                    labels: list of labels
    """
    texts_rows = pd.DataFrame(columns=texts.columns[1:])
    for index, text in texts.iterrows():
        start_index = 0
        fulltext = text.fulltext
        fulltext_split_by_rows = fulltext.split(split_by)
        for row in fulltext_split_by_rows:
            number_of_words_in_row = len(row.split())
            if number_of_words_in_row == 0:
                continue
            next_free = texts_rows.shape[0]
            texts_rows.loc[next_free] = [text.labels[start_index:start_index+number_of_words_in_row], text.data[start_index:start_index+number_of_words_in_row]]
            start_index = start_index + number_of_words_in_row
    return texts_rows


if __name__ == "__main__":
    # Load and create a dataframe with the data and the labels
    texts = generate_data()

    # Drop unnecessary columns
    texts = texts.drop(columns=['id', 'name', 'user_id', 'corpus_id', 'genre_id', 'length'])

    # Split the data by rows
    texts_rows = split_by_rows(texts)

    # Split the data by rows
    texts_paragraphs = split_by_rows(texts, split_by="\t\t\t")

    # Save the dataframe to a json file
    train_paragraphs, test_paragraphs = train_test_split(texts_rows, test_size=0.2)
    train_paragraphs, validation_paragraphs = train_test_split(train_paragraphs, test_size=0.2)
    train_paragraphs.to_json("data/train_paragraphs.json")
    test_paragraphs.to_json("data/test_paragraphs.json")
    validation_paragraphs.to_json("data/validation_paragraphs.json")

    # Save the dataframe to a json file
    train_rows, test_rows = train_test_split(texts_rows, test_size=0.2)
    train_rows, validation_rows = train_test_split(train_rows, test_size=0.2)
    train_rows.to_json("data/train_rows.json")
    test_rows.to_json("data/test_rows.json")
    validation_rows.to_json("data/validation_rows.json")

    # Save the dataframe to a json file
    train, test = train_test_split(texts, test_size=0.2)
    train, validation = train_test_split(train, test_size=0.2)
    train.to_json("data/train_full.json")
    test.to_json("data/test_full.json")
    validation.to_json("data/validation_full.json")
