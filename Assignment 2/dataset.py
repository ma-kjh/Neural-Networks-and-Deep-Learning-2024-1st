
# import some packages you need here
import os
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        
        # write your codes here
        self.data_dir = data_dir
        self.img_labels = os.listdir(self.data_dir)
        self.img_labels.sort()
        self.labels = list(map(lambda x : int(x[6]), self.img_labels))
        
        ## LeNet-5 input size 32x32
        ## train - data augmentation
        if 'train' in self.data_dir:
            ## same test
            self.transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.1307,), std = (0.3081,)) 
            ])
            
            ## augmented version
            # self.transform = transforms.Compose([
                # transforms.Resize((32,32)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean = (0.1307,), std = (0.3081,)) 
            # ])            
            
        else:
            self.transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.1307,), std = (0.3081,)) 
            ])

    def __len__(self):

        # write your codes here
        
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        # write your codes here
        img_path = os.path.join(self.data_dir, self.img_labels[idx])
        img = self.transform(Image.open(img_path))
        label = self.labels[idx]
        
        return img, label

if __name__ == '__main__':
    mnist_dataset = MNIST('/data/MNIST_assignment/train')
    print(mnist_dataset.__len__())
    print(mnist_dataset.__getitem__(6)[0].shape)
    # write test codes to verify your implementations


