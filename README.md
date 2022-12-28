# NLP-Project-Understanding-Figurative-Language-Hebrew

In this project we aim to predict metaphors from Hebrew Songs.

Data:
187 labeled texts in Hebrew (songs)

Data exploration:

Example: (Non-metaphors are not colored. The first word of a metaphor is colored in light Green, and the rest in green.
![alt text](data-exploration/Example_1.png)

Mataphor words frequency:

![alt text](data-exploration/metaphorWords.png)

We split the texts into rows and get 6379 samples.
Each word is tagged as a metaphor or non-metaphor word. (So we can insert the entire example into AlephBERT while fine-tuning)

We can see from the graph bellow that most of the rows has at least one metaphore.
62% of the data have at least one metaphor label.
![alt text](data-exploration/metaphorCount.png)

The data is imbalanced. 84.6% of the words are non-metaphors.


![alt text](data-exploration/LabelDistribution.png)

The following graphs show the most frequent words with each label:
![alt text](data-exploration/OFreq.png)
![alt text](data-exploration/IFreq.png)
![alt text](data-exploration/BFreq.png)

First results:
At the first stage I explored the following hyperparameters to find the best model:
- Ignoring the predictions for sub-tokens
- Learning rate
- Batch size
- Number of epochs for training
- Binary tagging / Three tags (0, B-metaphor, I-metaphor)
- Split by paragraphs / lines

Training:
Binary tags get the lowest loss on training. 
Batch Size from 32 and higher is not so noisy.

![Train_loss](https://user-images.githubusercontent.com/49562866/155875181-626f90c1-5177-4619-96a6-0834457773bd.png)

Eval loss is going down at start. Then it grows.
Experiments show that it's not helpful to stop at minimum or to use low learning rate.

![Eval_loss](https://user-images.githubusercontent.com/49562866/155875223-6289c4b4-81ae-4ea2-a156-8982f2d8c93a.png)

The best accuracy and f1 equared by the model that is trained on the binary classification.

![Eval_acc](https://user-images.githubusercontent.com/49562866/155875218-379708e2-7546-4624-93f4-173f74fe9d41.png)
![Eval_f1](https://user-images.githubusercontent.com/49562866/155875331-ea069d50-33e4-4d84-b5be-3fc4418f39e2.png)


After each experiment I did an evaluation on the validation set and on two more external corpus.
The results are shown bellow:
Validation:
![accuracy_by_model_validation_acc_2022-02-27_11-28-33](https://user-images.githubusercontent.com/49562866/155876958-b3a28829-fb4a-4c89-8f87-170c27372849.png)

![f1_by_model_validation_f1_2022-02-27_11-28-34](https://user-images.githubusercontent.com/49562866/155876973-e0ff84fa-1c2f-4ac4-8299-76b04d439705.png)

![f1_by_model (validation)2022-02-27_11-28-36](https://user-images.githubusercontent.com/49562866/155876963-e9d29f72-2dac-4722-835a-8a471b40d9a0.png)

External dataset 1:
![accuracy_by_model_TestDataset2022-02-27_11-28-40](https://user-images.githubusercontent.com/49562866/155876988-a6694a88-08e1-4740-899a-4168cdf6944b.png)

External dataset 2:
![accuracy_by_model_DocxDataset2022-02-27_11-28-37](https://user-images.githubusercontent.com/49562866/155877004-6db535f3-9228-4022-8dc4-ba312fc14945.png)

![f1_by_model_DocxDataset2022-02-27_11-28-39](https://user-images.githubusercontent.com/49562866/155877006-0572fb7c-aa93-40f4-830a-70aeeda70296.png)


We can see from the results:
- The accuracy of the models on validations get up to 94%.
- Increasing the number of epochs helps (even though the loss on validation is going up)
- Batch size of 64 and higher give almost the same results.
- F1 of unseen words is lower that the f1 on seen words
- Performance on external corpus drops dramatically (81% accuracy)


