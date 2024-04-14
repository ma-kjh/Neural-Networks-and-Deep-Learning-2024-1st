## MNIST Classification


In this Assignment, We build a neural network classifier with MNIST dataset.


### Get Started

`dataset.py, model.py, main.py , README.md (Report) files`

```
├── ...
├── results
│   ├── checkpoints
│   └── plots
├── dataset.py
├── model.py
├── main.py
```

### Data

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. You can download [here](http://yann.lecun.com/exdb/mnist/)

Our codebase accesses the datasets from `/data/MNIST_assignment/` by default.

### Models

- LeNet5 : # of Params - 61,706
- CustomMLP : # of Params - 62,170


### Report 1

```
Implement LeNet-5 and your custom MLP models in model.py.

Some instructions are given in the file as comments.

Note that your custom MLP model should have about the same number of model parameters with LeNet-5.

Describe the number of model parameters of LeNet-5 and your custom MLP and how to compute them in your report.

Write main.py to train your models, LeNet-5 and custom MLP.

Here, you should monitor the training process.

To do so, you need some statistics such as average loss values and accuracy at the end of each epoch.
```

#### LeNet5

![Screenshot 2024-04-14 at 16 06 44](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/f0e5d3c0-9ada-478f-9aab-34c143b88cb6)

`In our case, we use input size (28,28,1).`

```python
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)  # # of params : (5 * 5 * 1 * 6) + (1 * 6) = 156
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1) # # of params : (5 * 5 * 6 * 16) + (1 * 16) = 2,416
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1) # # of params : (5 * 5 * 16 * 120) + (1 * 120) = 48,120
        
        self.subsampling = nn.MaxPool2d(kernel_size=(2,2), stride=None) 
        
        self.fc1 = nn.Linear(120, 84) # # of params : (120 * 84) + (1 * 84) = 10,164
        self.fc2 = nn.Linear(84, 10) # # of params : (84 * 10) + (1 * 10) = 850
        
        ## total # of params : 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706
```

#### CustomMLP

```python
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64) # # of params : 50,176 + 64 = 50,240
        self.fc2 = nn.Linear(64, 64) # # of params : 4,096 + 64 = 4,160
        self.fc3 = nn.Linear(64, 64) # # of params : 4,096 + 64 = 4,160
        self.fc4 = nn.Linear(64, 48) # # of params : 3,072 + 48 = 3,120
        self.fc5 = nn.Linear(48, 10) # # of params : 480 + 10 = 490
        
        ## total # of params : 50,240 + 4,160 + 4,160 + 3,120 + 490 = 62,170
```


### (Report 1)

Implement LeNet-5 and your custom MLP models in model.py. Some instructions are given in the file as comments. Note that your custom MLP model should have about the same number of model parameters with LeNet-5. Describe the number of model parameters of LeNet-5 and your custom MLP and how to compute them in your report.
Write main.py to train your models, LeNet-5 and custom MLP. Here, you should monitor the training process. To do so, you need some statistics such as average loss values and accuracy at the end of each epoch.







(Report) Plot above statistics, average loss value and accuracy, for training and testing. It is fine to use the test dataset as a validation dataset. Therefore, you will have four plots for each model: loss and accuracy curves for training and test datasets, respectively.

(Report) Compare the predictive performances of LeNet-5 and your custom MLP. Also, make sure that the accuracy of LeNet-5 (your implementation) is similar to the known accuracy. 

(Report) Employ at least more than two regularization techniques to improve LeNet-5 model. You can use whatever techniques if you think they may be helpful to improve the performance. Verify that they actually help improve the performance. Keep in mind that when you employ the data augmentation technique, it should be applied only to training data. So, the modification of provided MNIST class in dataset.py may be needed.

`Note that the details of training configuration which are not mentioned in this document and the comments can be defined yourself. For example, decide how many epochs you will train the model.`
