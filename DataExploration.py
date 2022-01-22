from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import docx
from docx.enum.text import WD_COLOR_INDEX

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

label_names = {"O": 0, "B-metaphor": 1, "I-metaphor": 2}

raw_datasets = load_dataset('MetaphorDataset.py')
train = raw_datasets['train']
test = raw_datasets['test']
validation = raw_datasets['validation']

# Calculate statistics
label_hist = np.zeros([3])
word_dict = {}
examples_with_metaphor = np.zeros([3])
for example in train:
    bool_metaphor = np.zeros([3])
    words = example['data']
    labels = example['labels']
    for label, word in zip(labels, words):
        if word not in word_dict:
            word_dict[word] = np.zeros([3])
            word_dict[word][label] = 1
        else:
            word_dict[word][label] += 1
        label_hist[label] += 1
        bool_metaphor[label] = True
    for index, is_label in enumerate(bool_metaphor):
        examples_with_metaphor[index] += int(is_label)


# Plot statistics
plt.bar(range(3), label_hist)
# show numbers on the bars
for i in range(3):
    plt.text(i, label_hist[i], str(label_hist[i]))
# change sticks names
plt.xticks(range(3), [label for label in label_names])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.show()


# Number of examples with at least one metaphor
plt.figure(figsize=(10,5))
plt.bar(range(3), examples_with_metaphor, align='center')
# show numbers on the bars
for i in range(3):
    plt.text(i, examples_with_metaphor[i], str(examples_with_metaphor[i]))
plt.xticks(range(3), ['No metaphor', 'B', 'I'])
plt.xlabel('Label')
plt.ylabel('Number of examples')
plt.title('Number of examples with / without at least one label')
plt.show()

# create a dataframe from the dictionary
df = pd.DataFrame.from_dict(word_dict, orient='index')
df.rename(columns={0: "O", 1: "B", 2: "I"}, inplace=True)
df.index.name = 'word'

# Plot top 10 words with highest frequency of O label
plt.figure(figsize=(10, 10))
df.sort_values('O', ascending=False).head(10).plot.bar(y=['O', 'I', 'B'], legend=True, stacked=True)
# Reverse words in sticks ( end to start because this is hebrew )
reversed_sticks = []
for stick in plt.xticks()[1]:
    reversed_sticks.append(stick.get_text()[::-1])
plt.xticks(range(len(reversed_sticks)), reversed_sticks)
plt.title('Top 10 words with highest frequency of O label')
plt.ylabel('Frequency')
plt.show()

# Plot top 10 words with highest frequency of B label
plt.figure(figsize=(10, 10))
df.sort_values('B', ascending=False).head(10).plot.bar(y=['O', 'I', 'B'], legend=True, stacked=True)
# Reverse words in sticks ( end to start because this is hebrew )
reversed_sticks = []
for stick in plt.xticks()[1]:
    reversed_sticks.append(stick.get_text()[::-1])
plt.xticks(range(len(reversed_sticks)), reversed_sticks)
plt.title('Top 10 words with highest frequency of B label')
plt.ylabel('Frequency')
plt.show()

# Plot top 10 words with highest frequency of I label
plt.figure(figsize=(10, 10))
df.sort_values('I', ascending=False).head(10).plot.bar(y=['O', 'I', 'B'], legend=True,
                                                       stacked=True)
# Reverse words in sticks ( end to start because this is hebrew )
reversed_sticks = []
for stick in plt.xticks()[1]:
    reversed_sticks.append(stick.get_text()[::-1])
plt.xticks(range(len(reversed_sticks)), reversed_sticks)
plt.title('Top 10 words with highest frequency of I label')
plt.ylabel('Frequency')
plt.show()

# Print 20 examples with its labels
doc = docx.Document()
doc.add_heading('Words with metaphors highlighted', 0)
random_indices = np.random.choice(range(len(train)), 20, replace=False)
for index in random_indices:
    example = train[int(index)]
    words = example['data']
    labels = example['labels']
    para = doc.add_paragraph('Example: ' + str(index) + '\n')
    for word, label in zip(words, labels):
        if label == 0:
            para.add_run(word + ' ')
        elif label == 1:
            para.add_run(word + ' ').font.highlight_color = docx.enum.text.WD_COLOR_INDEX.YELLOW
        else:  # label == 2
            para.add_run(word + ' ').font.highlight_color = docx.enum.text.WD_COLOR_INDEX.GREEN

    # Now save the document to a location
    doc.save('examples.docx')




