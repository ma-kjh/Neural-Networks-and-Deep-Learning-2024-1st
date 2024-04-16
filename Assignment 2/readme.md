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

### Script
`CUDA_VISIBLE_DEIVCES=# python main.py`

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
CUSTOMMLP TRAIN AND TEST
TRAIN START!
1|20 - 5.0 % finished
1|20 - TRAIN LOSS : 1.9639
1|20 - TRAIN ACC : 0.5162
1|20 - VAL LOSS : 1.6972
1|20 - VAL ACC : 0.7639
2|20 - 10.0 % finished
2|20 - TRAIN LOSS : 1.6283
2|20 - TRAIN ACC : 0.8386
2|20 - VAL LOSS : 1.6144
2|20 - VAL ACC : 0.8495
3|20 - 15.0 % finished
3|20 - TRAIN LOSS : 1.5597
3|20 - TRAIN ACC : 0.9060
3|20 - VAL LOSS : 1.5331
3|20 - VAL ACC : 0.9307
4|20 - 20.0 % finished
4|20 - TRAIN LOSS : 1.5271
4|20 - TRAIN ACC : 0.9364
4|20 - VAL LOSS : 1.5278
4|20 - VAL ACC : 0.9339
5|20 - 25.0 % finished
5|20 - TRAIN LOSS : 1.5210
5|20 - TRAIN ACC : 0.9420
5|20 - VAL LOSS : 1.5188
5|20 - VAL ACC : 0.9433
6|20 - 30.0 % finished
6|20 - TRAIN LOSS : 1.5155
6|20 - TRAIN ACC : 0.9470
6|20 - VAL LOSS : 1.5234
6|20 - VAL ACC : 0.9382
7|20 - 35.0 % finished
7|20 - TRAIN LOSS : 1.5148
7|20 - TRAIN ACC : 0.9471
7|20 - VAL LOSS : 1.5168
7|20 - VAL ACC : 0.9448
8|20 - 40.0 % finished
8|20 - TRAIN LOSS : 1.5135
8|20 - TRAIN ACC : 0.9484
8|20 - VAL LOSS : 1.5133
8|20 - VAL ACC : 0.9482
9|20 - 45.0 % finished
9|20 - TRAIN LOSS : 1.5117
9|20 - TRAIN ACC : 0.9502
9|20 - VAL LOSS : 1.5215
9|20 - VAL ACC : 0.9398
10|20 - 50.0 % finished
10|20 - TRAIN LOSS : 1.5090
10|20 - TRAIN ACC : 0.9524
10|20 - VAL LOSS : 1.5173
10|20 - VAL ACC : 0.9432
11|20 - 55.00000000000001 % finished
11|20 - TRAIN LOSS : 1.5071
11|20 - TRAIN ACC : 0.9542
11|20 - VAL LOSS : 1.5131
11|20 - VAL ACC : 0.9488
12|20 - 60.0 % finished
12|20 - TRAIN LOSS : 1.5081
12|20 - TRAIN ACC : 0.9530
12|20 - VAL LOSS : 1.5094
12|20 - VAL ACC : 0.9516
13|20 - 65.0 % finished
13|20 - TRAIN LOSS : 1.5044
13|20 - TRAIN ACC : 0.9570
13|20 - VAL LOSS : 1.5087
13|20 - VAL ACC : 0.9530
14|20 - 70.0 % finished
14|20 - TRAIN LOSS : 1.5054
14|20 - TRAIN ACC : 0.9564
14|20 - VAL LOSS : 1.5123
14|20 - VAL ACC : 0.9490
15|20 - 75.0 % finished
15|20 - TRAIN LOSS : 1.5045
15|20 - TRAIN ACC : 0.9569
15|20 - VAL LOSS : 1.5145
15|20 - VAL ACC : 0.9474
16|20 - 80.0 % finished
16|20 - TRAIN LOSS : 1.5037
16|20 - TRAIN ACC : 0.9578
16|20 - VAL LOSS : 1.5094
16|20 - VAL ACC : 0.9523
17|20 - 85.0 % finished
17|20 - TRAIN LOSS : 1.5035
17|20 - TRAIN ACC : 0.9580
17|20 - VAL LOSS : 1.5056
17|20 - VAL ACC : 0.9560
18|20 - 90.0 % finished
18|20 - TRAIN LOSS : 1.5038
18|20 - TRAIN ACC : 0.9573
18|20 - VAL LOSS : 1.5084
18|20 - VAL ACC : 0.9526
19|20 - 95.0 % finished
19|20 - TRAIN LOSS : 1.5030
19|20 - TRAIN ACC : 0.9584
19|20 - VAL LOSS : 1.5061
19|20 - VAL ACC : 0.9546
20|20 - 100.0 % finished
20|20 - TRAIN LOSS : 1.5028
20|20 - TRAIN ACC : 0.9587
20|20 - VAL LOSS : 1.5079
20|20 - VAL ACC : 0.9532
TRAIN END!
TEST FINISHED
TEST LOSS : 1.5079
TEST ACC : 0.9532
```

</div>
</details>

![Report2_CustomMLP_trn_val_com](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/f9fa7992-ff2c-4c7c-aaf0-9be0f19634ff)


![Report2_CustomMLP_trn_val_sep](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/9cb18487-ac4f-4304-bd4f-4df56246b102)



`First of all, we want to make the model work properly when dataset(test or val) that had not been seen in training will be received. However, as you can see, the gap of training and validation measurements gets bigger when epochs are increased. You should mitigate this effect of overfitting by using some regularization or other techniques.`



<details>
<summary> LeNet Reg trn val Log </summary>
<div>


```
LENET REG TRAIN AND TEST
TRAIN START!
1|20 - 5.0 % finished
1|20 - TRAIN LOSS : 1.5420
1|20 - TRAIN ACC : 0.9483
1|20 - VAL LOSS : 1.4865
1|20 - VAL ACC : 0.9813
2|20 - 10.0 % finished
2|20 - TRAIN LOSS : 1.4873
2|20 - TRAIN ACC : 0.9802
2|20 - VAL LOSS : 1.4794
2|20 - VAL ACC : 0.9854
3|20 - 15.0 % finished
3|20 - TRAIN LOSS : 1.4816
3|20 - TRAIN ACC : 0.9836
3|20 - VAL LOSS : 1.4764
3|20 - VAL ACC : 0.9883
4|20 - 20.0 % finished
4|20 - TRAIN LOSS : 1.4777
4|20 - TRAIN ACC : 0.9869
4|20 - VAL LOSS : 1.4755
4|20 - VAL ACC : 0.9886
5|20 - 25.0 % finished
5|20 - TRAIN LOSS : 1.4755
5|20 - TRAIN ACC : 0.9886
5|20 - VAL LOSS : 1.4731
5|20 - VAL ACC : 0.9909
6|20 - 30.0 % finished
6|20 - TRAIN LOSS : 1.4743
6|20 - TRAIN ACC : 0.9895
6|20 - VAL LOSS : 1.4727
6|20 - VAL ACC : 0.9907
7|20 - 35.0 % finished
7|20 - TRAIN LOSS : 1.4728
7|20 - TRAIN ACC : 0.9909
7|20 - VAL LOSS : 1.4724
7|20 - VAL ACC : 0.9906
8|20 - 40.0 % finished
8|20 - TRAIN LOSS : 1.4721
8|20 - TRAIN ACC : 0.9912
8|20 - VAL LOSS : 1.4723
8|20 - VAL ACC : 0.9908
9|20 - 45.0 % finished
9|20 - TRAIN LOSS : 1.4712
9|20 - TRAIN ACC : 0.9920
9|20 - VAL LOSS : 1.4722
9|20 - VAL ACC : 0.9905
10|20 - 50.0 % finished
10|20 - TRAIN LOSS : 1.4706
10|20 - TRAIN ACC : 0.9925
10|20 - VAL LOSS : 1.4718
10|20 - VAL ACC : 0.9907
11|20 - 55.00000000000001 % finished
11|20 - TRAIN LOSS : 1.4697
11|20 - TRAIN ACC : 0.9933
11|20 - VAL LOSS : 1.4716
11|20 - VAL ACC : 0.9913
12|20 - 60.0 % finished
12|20 - TRAIN LOSS : 1.4694
12|20 - TRAIN ACC : 0.9934
12|20 - VAL LOSS : 1.4737
12|20 - VAL ACC : 0.9891
13|20 - 65.0 % finished
13|20 - TRAIN LOSS : 1.4691
13|20 - TRAIN ACC : 0.9941
13|20 - VAL LOSS : 1.4709
13|20 - VAL ACC : 0.9916
14|20 - 70.0 % finished
14|20 - TRAIN LOSS : 1.4688
14|20 - TRAIN ACC : 0.9938
14|20 - VAL LOSS : 1.4708
14|20 - VAL ACC : 0.9910
15|20 - 75.0 % finished
15|20 - TRAIN LOSS : 1.4683
15|20 - TRAIN ACC : 0.9945
15|20 - VAL LOSS : 1.4714
15|20 - VAL ACC : 0.9916
16|20 - 80.0 % finished
16|20 - TRAIN LOSS : 1.4678
16|20 - TRAIN ACC : 0.9948
16|20 - VAL LOSS : 1.4705
16|20 - VAL ACC : 0.9924
17|20 - 85.0 % finished
17|20 - TRAIN LOSS : 1.4674
17|20 - TRAIN ACC : 0.9952
17|20 - VAL LOSS : 1.4706
17|20 - VAL ACC : 0.9921
18|20 - 90.0 % finished
18|20 - TRAIN LOSS : 1.4674
18|20 - TRAIN ACC : 0.9951
18|20 - VAL LOSS : 1.4701
18|20 - VAL ACC : 0.9923
19|20 - 95.0 % finished
19|20 - TRAIN LOSS : 1.4666
19|20 - TRAIN ACC : 0.9959
19|20 - VAL LOSS : 1.4706
19|20 - VAL ACC : 0.9917
20|20 - 100.0 % finished
20|20 - TRAIN LOSS : 1.4664
20|20 - TRAIN ACC : 0.9961
20|20 - VAL LOSS : 1.4705
20|20 - VAL ACC : 0.9914
TRAIN END!
TEST FINISHED
TEST LOSS : 1.4705
TEST ACC : 0.9914
```


</div>
</details>

![Report2_LeNet5_reg_trn_val_com](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/6607068f-9a18-4a2b-9b94-5634140d2526)


![Report2_LeNet5_reg_trn_val_sep](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/388dd726-9d77-4775-aee9-97dfce83cc2c)


`In our case, we emply two regularization techniques, L2 Regularization (SGD weight decay) and BatchNorm. As you can see, L2 Reg and BatchNorm can improve little bit the validation performance than original one.`

