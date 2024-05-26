# import some packages you need here
import numpy as np
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
			You need this dictionary to generate characters.
		2) Make list of character indices using the dictionary
		3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        ### seq_lenth = 30 ###
        seq_length=30
        ######################
        with open(f'{input_file}','r',encoding='utf-8') as file:
            self.text = file.read()
        
        chars = sorted(list(set(self.text)))
        self.char_to_idx = {c : i for i, c in enumerate(chars)}
        self.idx_to_char = {i : c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_length = seq_length

        self.data = [self.char_to_idx[char] for char in self.text]


    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):

        input = self.data[idx: idx + self.seq_length]
        target = self.data[idx + 1:idx + self.seq_length + 1]
        
        return torch.tensor(input, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# if __name__ == '__main__':
    # train_dataset = Shakespeare('./data/shakespeare_train.txt')
    # print(train_dataset[0])