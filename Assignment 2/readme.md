## MNIST Classification


In this Assignment, We build a neural network classifier with MNIST dataset.


### Get Started

`dataset.py, model.py, main.py , README.md (Report) files`

```
├── ...
├── results
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
- CustomMLP : # of Params - 61,450

#### LeNet5

![Screenshot 2024-04-14 at 16 06 44](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/f0e5d3c0-9ada-478f-9aab-34c143b88cb6)

`In this case, transform input size (28,28,1) to (32,32,1) because I want to follow original LeNet5 paper.`

```python
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)  # # of params : (5 * 5 * 1 * 6) + (1 * 6) = 156
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1) # # of params : (5 * 5 * 6 * 16) + (1 * 16) = 2,416
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1) # # of params : (5 * 5 * 16 * 120) + (1 * 120) = 48,120
        
        self.subsampling = nn.MaxPool2d(kernel_size=(2,2), stride=None) 
        
        self.activation = nn.Tanh()
        
        self.fc1 = nn.Linear(120, 84) # # of params : (120 * 84) + (1 * 84) = 10,164
        self.fc2 = nn.Linear(84, 10) # # of params : (84 * 10) + (1 * 10) = 850
        
        ## total # of params : 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706
```

#### CustomMLP

```python
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 48) # # of params : (32 * 32 * 1 * 48) + 48 = 49,152 + 48 = 49,200
        self.fc2 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc3 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc4 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc5 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc6 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc7 = nn.Linear(48, 10) # # of params : 480 + 10 = 490
        
        self.activation = nn.Tanh()
        
        ## total # of params : 49,200 + 2,352 + 2,352 + 2,352 + 2,352 + 2,352 + 490 = 61,450
```

LeNet5 has 61,706 parameters and CustomMLP has 61,450 parameters. They have similar parameters.


### Report 2


#### LeNet5 trn val

```
12|12 - TRAIN LOSS : 1.4692
12|12 - TRAIN ACC : 0.9934
12|12 - VAL LOSS : 1.4740
12|12 - VAL ACC : 0.9888
```

![Report2_LeNet5_trn_val_sep](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/ceb7ab5a-a388-4cef-ab7b-45424a88fcad)


![Report2_LeNet5_trn_val_com](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/9a96b543-c13a-40b6-a440-c6f157a3cfc9)




#### CustomMLP trn val
```
12|12 - TRAIN LOSS : 1.5100
12|12 - TRAIN ACC : 0.9517
12|12 - VAL LOSS : 1.5127
12|12 - VAL ACC : 0.9489
```

![Report2_CustomMLP_trn_val_sep](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/753dd158-0a20-4ff6-a90c-f8de6b513e1f)

![Report2_CustomMLP_trn_val_com](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/740a1c4d-f69d-4d89-a52a-c911f3c3791f)




### Report 3

```
Employ at least more than two regularization techniques to improve LeNet-5 model.

You can use whatever techniques if you think they may be helpful to improve the performance.

Verify that they actually help improve the performance.

Keep in mind that when you employ the data augmentation technique, it should be applied only to training data.

So, the modification of provided MNIST class in dataset.py may be needed.
```

`In our case, we emply two regularization techniques, L2 Regularization (SGD weight decay) and BatchNorm.`

`As you can see, L2 Reg and BatchNorm can improve little bit the validation performance than original one.`



`LeNet5-Reg trn val`
```
12|12 - TRAIN LOSS : 1.4694
12|12 - TRAIN ACC : 0.9936
12|12 - VAL LOSS : 1.4730
12|12 - VAL ACC : 0.9894 
```

![Report2_LeNet5_trn_val_sep](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/d6a1da9c-a682-41b8-a02d-230d3f17244a)

![Report2_LeNet5_trn_val_com](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/ad28a8b7-ed98-47cb-8211-08d46e60b7aa)


