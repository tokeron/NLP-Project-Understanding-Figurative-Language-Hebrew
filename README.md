# NLP-Project-Understanding-Figurative-Language-Hebrew

In this project we aim to predicr metaphors from Hebrew Songs.

Data:
187 labeled texts in Hebrew (songs)

Data exploration:

Examples:
![alt text](./DataExploration/Example_1.png)

We split the data into rows (So we can insert the entire example into AlephBERT while fine-tuning)
![alt text](./DataExploration/ExampleSplit_2.png)

We can see from the graph bellow that most of the rows has at least one metaphore.
![alt text](./DataExploration/metaphorCount.png)

We can see from the graph bellow that the data is imbalanced.
![alt text](./DataExploration/LabelDistribution.png)

The following graphs show the most frequent words with each label:
![alt text](./DataExploration/OFreq.png)
![alt text](./DataExploration/IFreq.png)
![alt text](./DataExploration/BFreq.png)
