
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self, regularization=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)  # # of params : (5 * 5 * 1 * 6) + (1 * 6) = 156
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1) # # of params : (5 * 5 * 6 * 16) + (1 * 16) = 2,416
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1) # # of params : (5 * 5 * 16 * 120) + (1 * 120) = 48,120
        
        self.subsampling = nn.MaxPool2d(kernel_size=(2,2), stride=None) 
        
        self.activation = nn.Tanh()
        
        self.fc1 = nn.Linear(120, 84) # # of params : (120 * 84) + (1 * 84) = 10,164
        self.fc2 = nn.Linear(84, 10) # # of params : (84 * 10) + (1 * 10) = 850
        
        ## total # of params : 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706
        # write your codes here
        
        self.regularization = regularization
        
        ## Conv
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(120)
        
        ## Linear
        self.bn4 = nn.BatchNorm1d(84)

    def forward(self, img):
        x = self.conv1(img) # Bx6x28x28
        
        if self.regularization:
            x = self.bn1(x)
        
        x = self.activation(x)
        
        x = self.subsampling(x) # Bx6x14x14
        
        x = self.conv2(x) # Bx16x10x10
        
        if self.regularization:
            x = self.bn2(x)
            
        
        x = self.activation(x)
        
        x = self.subsampling(x) # Bx6x5x5
        
        x = self.conv3(x) # Bx120x1x1
        
        if self.regularization:
            x = self.bn3(x)
        
        x = self.activation(x)
        
        x = x.view(-1,120) # Bx120
        
        x = self.fc1(x) # Bx84
        
        if self.regularization:
            x = self.bn4(x)
        
        x = self.activation(x)
        
        x = self.fc2(x) # Bx10
        
        output = F.softmax(x, dim=1)
        # write your codes here

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 48) # # of params : (1024 * 48) + 48 = 49,152 + 48 = 49,200
        self.fc2 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc3 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc4 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc5 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc6 = nn.Linear(48, 48) # # of params : (48 * 48) + 48 = 2,304 + 48 = 2,352
        self.fc7 = nn.Linear(48, 10) # # of params : 480 + 10 = 490
        
        self.activation = nn.Tanh()
        
        ## total # of params : 49,200 + 2,352 + 2,352 + 2,352 + 2,352 + 2,352 + 490 = 61,450
        
        # write your codes here

    def forward(self, img):
        x = img.view(-1, 1024)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        x = self.activation(x)
        x = self.fc6(x)
        x = self.activation(x)
        x = self.fc7(x)
        output = F.softmax(x, dim=1)
        # write your codes here

        return output

# if __name__ == '__main__':
#     from torchsummary import summary
    
#     c = CustomMLP().cuda()
#     s = LeNet5().cuda()
#     summary(c, (1,32,32))
#     summary(s, (1,32,32))
    