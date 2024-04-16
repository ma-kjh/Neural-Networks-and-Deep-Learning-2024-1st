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


### Experiments

<details>
<summary> LeNet5 trn val Log </summary>
<div>


```
LENET TRAIN AND TEST
TRAIN START!
1|20 - 5.0 % finished
1|20 - TRAIN LOSS : 1.7391
1|20 - TRAIN ACC : 0.7743
1|20 - VAL LOSS : 1.5216
1|20 - VAL ACC : 0.9496
2|20 - 10.0 % finished
2|20 - TRAIN LOSS : 1.5057
2|20 - TRAIN ACC : 0.9630
2|20 - VAL LOSS : 1.4932
2|20 - VAL ACC : 0.9734
3|20 - 15.0 % finished
3|20 - TRAIN LOSS : 1.4894
3|20 - TRAIN ACC : 0.9768
3|20 - VAL LOSS : 1.4837
3|20 - VAL ACC : 0.9820
4|20 - 20.0 % finished
4|20 - TRAIN LOSS : 1.4828
4|20 - TRAIN ACC : 0.9822
4|20 - VAL LOSS : 1.4810
4|20 - VAL ACC : 0.9831
5|20 - 25.0 % finished
5|20 - TRAIN LOSS : 1.4790
5|20 - TRAIN ACC : 0.9852
5|20 - VAL LOSS : 1.4809
5|20 - VAL ACC : 0.9826
6|20 - 30.0 % finished
6|20 - TRAIN LOSS : 1.4768
6|20 - TRAIN ACC : 0.9870
6|20 - VAL LOSS : 1.4774
6|20 - VAL ACC : 0.9860
7|20 - 35.0 % finished
7|20 - TRAIN LOSS : 1.4745
7|20 - TRAIN ACC : 0.9891
7|20 - VAL LOSS : 1.4761
7|20 - VAL ACC : 0.9872
8|20 - 40.0 % finished
8|20 - TRAIN LOSS : 1.4731
8|20 - TRAIN ACC : 0.9902
8|20 - VAL LOSS : 1.4765
8|20 - VAL ACC : 0.9867
9|20 - 45.0 % finished
9|20 - TRAIN LOSS : 1.4718
9|20 - TRAIN ACC : 0.9914
9|20 - VAL LOSS : 1.4754
9|20 - VAL ACC : 0.9875
10|20 - 50.0 % finished
10|20 - TRAIN LOSS : 1.4709
10|20 - TRAIN ACC : 0.9919
10|20 - VAL LOSS : 1.4754
10|20 - VAL ACC : 0.9874
11|20 - 55.00000000000001 % finished
11|20 - TRAIN LOSS : 1.4700
11|20 - TRAIN ACC : 0.9927
11|20 - VAL LOSS : 1.4763
11|20 - VAL ACC : 0.9856
12|20 - 60.0 % finished
12|20 - TRAIN LOSS : 1.4693
12|20 - TRAIN ACC : 0.9936
12|20 - VAL LOSS : 1.4745
12|20 - VAL ACC : 0.9877
13|20 - 65.0 % finished
13|20 - TRAIN LOSS : 1.4686
13|20 - TRAIN ACC : 0.9939
13|20 - VAL LOSS : 1.4738
13|20 - VAL ACC : 0.9890
14|20 - 70.0 % finished
14|20 - TRAIN LOSS : 1.4678
14|20 - TRAIN ACC : 0.9947
14|20 - VAL LOSS : 1.4748
14|20 - VAL ACC : 0.9876
15|20 - 75.0 % finished
15|20 - TRAIN LOSS : 1.4673
15|20 - TRAIN ACC : 0.9949
15|20 - VAL LOSS : 1.4739
15|20 - VAL ACC : 0.9882
16|20 - 80.0 % finished
16|20 - TRAIN LOSS : 1.4669
16|20 - TRAIN ACC : 0.9953
16|20 - VAL LOSS : 1.4739
16|20 - VAL ACC : 0.9880
17|20 - 85.0 % finished
17|20 - TRAIN LOSS : 1.4663
17|20 - TRAIN ACC : 0.9960
17|20 - VAL LOSS : 1.4732
17|20 - VAL ACC : 0.9888
18|20 - 90.0 % finished
18|20 - TRAIN LOSS : 1.4660
18|20 - TRAIN ACC : 0.9961
18|20 - VAL LOSS : 1.4736
18|20 - VAL ACC : 0.9885
19|20 - 95.0 % finished
19|20 - TRAIN LOSS : 1.4658
19|20 - TRAIN ACC : 0.9962
19|20 - VAL LOSS : 1.4732
19|20 - VAL ACC : 0.9888
20|20 - 100.0 % finished
20|20 - TRAIN LOSS : 1.4656
20|20 - TRAIN ACC : 0.9964
20|20 - VAL LOSS : 1.4741
20|20 - VAL ACC : 0.9876
TRAIN END!
TEST FINISHED
TEST LOSS : 1.4741
TEST ACC : 0.9876
```

</div>
</details>

![Report2_LeNet5_trn_val_com](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/0094e4ad-8eb5-4def-98fe-ae713f61bf97)


![Report2_LeNet5_trn_val_sep](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/09f1f987-60d5-4e17-87f5-3b6f706df120)


#### CustomMLP trn val

<details>
<summary> CustomMLP trn val Log </summary>
<div>


```
LENET TRAIN AND TEST
TRAIN START!
1|20 - 5.0 % finished
1|20 - TRAIN LOSS : 1.7266
1|20 - TRAIN ACC : 0.7801
1|20 - VAL LOSS : 1.5113
1|20 - VAL ACC : 0.9584
2|20 - 10.0 % finished
2|20 - TRAIN LOSS : 1.5026
2|20 - TRAIN ACC : 0.9655
2|20 - VAL LOSS : 1.4890
2|20 - VAL ACC : 0.9766

```

</div>
</details>


![Report2_CustomMLP_trn_val_sep](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/753dd158-0a20-4ff6-a90c-f8de6b513e1f)

![Report2_CustomMLP_trn_val_com](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/740a1c4d-f69d-4d89-a52a-c911f3c3791f)

`First of all, as you can see, the gap of training and validation measurements is similar in early training process and even better in validation datasets, this is called underfitting. However, It gets bigger when epochs are increased. This is called overfitting for training datasets, and you can fix this problem by using regularization or other techniques.`


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


