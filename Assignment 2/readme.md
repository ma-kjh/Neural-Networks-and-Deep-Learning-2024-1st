## MNIST Classification

In this assignment, you build a neural network classifier with MNIST dataset. For a detailed description about MNIST dataset, please refer to this link.

Due date: 2024. 04. 16. Tue 23:59

Submission: Personal Github repo URL

`dataset.py, model.py, main.py , README.md (Report) files`

`Total score: High/Medium/Low`

Requirements

You should write your own pipeline to provide data to your model. Write your code in the template dataset.py. 

Please read the comments carefully and follow those instructions.

(Report) Implement LeNet-5 and your custom MLP models in model.py. Some instructions are given in the file as comments. Note that your custom MLP model should have about the same number of model parameters with LeNet-5. Describe the number of model parameters of LeNet-5 and your custom MLP and how to compute them in your report.
Write main.py to train your models, LeNet-5 and custom MLP. Here, you should monitor the training process. To do so, you need some statistics such as average loss values and accuracy at the end of each epoch.

(Report) Plot above statistics, average loss value and accuracy, for training and testing. It is fine to use the test dataset as a validation dataset. Therefore, you will have four plots for each model: loss and accuracy curves for training and test datasets, respectively.

(Report) Compare the predictive performances of LeNet-5 and your custom MLP. Also, make sure that the accuracy of LeNet-5 (your implementation) is similar to the known accuracy. 

(Report) Employ at least more than two regularization techniques to improve LeNet-5 model. You can use whatever techniques if you think they may be helpful to improve the performance. Verify that they actually help improve the performance. Keep in mind that when you employ the data augmentation technique, it should be applied only to training data. So, the modification of provided MNIST class in dataset.py may be needed.

`Note that the details of training configuration which are not mentioned in this document and the comments can be defined yourself. For example, decide how many epochs you will train the model.`
