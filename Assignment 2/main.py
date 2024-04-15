
import dataset
from model import LeNet5, CustomMLP

import torch
import numpy as np
import matplotlib.pyplot as plt
# import some packages you need here

def plotting(trn_val_loss, trn_val_acc, modelname='LeNet5'):

    trn_loss = trn_val_loss[0]
    val_loss = trn_val_loss[1]
    
    trn_acc = trn_val_acc[0]
    val_acc = trn_val_acc[1]
    
    x = list(range(1, len(trn_loss)+1))

    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(x, trn_loss, label='trn_loss')
    plt.plot(x, val_loss, label='val_loss')
    plt.title(f"{modelname} trn val loss")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(x, trn_acc, label='trn_acc')
    plt.plot(x, val_acc, label='val_acc')
    plt.title(f"{modelname} trn val acc")
    plt.legend()
    
    
    plt.savefig(f"./results/plots/Report2_{modelname}_trn_val_com.png")
    
    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    plt.plot(trn_loss, label='trn_loss')
    plt.title(f"{modelname} trn loss")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(val_loss, label='val_loss')
    plt.title(f"{modelname} val loss")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(trn_acc, label='trn_acc')
    plt.title(f"{modelname} trn acc")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(val_acc, label='val_acc')
    plt.title(f"{modelname} val acc")
    plt.legend()
    
    plt.savefig(f"./results/plots/Report2_{modelname}_trn_val_sep.png")
    

    return None

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    #
    model.to(device)
    
    ### Train ###
    trn_loss = []
    trn_acc = []
    
    tst_loss = []
    tst_acc = []

    tst_loader = trn_loader[1]
    trn_loader = trn_loader[0]
    print(f"TRAIN START!")
    for epoch in range(1, EPOCHS+1):
        model.train()
        trn_loss_epoch = 0
        trn_acc_epoch = 0
        
        for i, (batch) in enumerate(trn_loader):
            optimizer.zero_grad()
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            trn_loss_epoch += loss
            trn_acc_epoch += torch.sum(torch.argmax(output,dim=1) == labels)
            
        print(f"{epoch}|{EPOCHS} - {epoch/EPOCHS * 100} % finished")
        print(f"{epoch}|{EPOCHS} - TRAIN LOSS : {trn_loss_epoch/len(trn_loader):.4f}") # exactly, train batch loss mean
        print(f"{epoch}|{EPOCHS} - TRAIN ACC : {trn_acc_epoch/len(trn_loader.dataset):.4f}")
        trn_loss.append(trn_loss_epoch.detach().cpu().numpy()/len(trn_loader))
        trn_acc.append(trn_acc_epoch.detach().cpu().numpy()/len(trn_loader.dataset))
        
        
        ### Validation ###
        tst_loss_epoch = 0
        tst_acc_epoch = 0
        
        model.eval()
        
        with torch.no_grad():
            for i, (batch) in enumerate(tst_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                output = model(images)
                loss = criterion(output, labels)
                tst_loss_epoch += loss
                tst_acc_epoch += torch.sum(torch.argmax(output,dim=1) == labels)
        print(f"{epoch}|{EPOCHS} - VAL LOSS : {tst_loss_epoch/len(tst_loader):.4f}") # exactly, train batch loss mean
        print(f"{epoch}|{EPOCHS} - VAL ACC : {tst_acc_epoch/len(tst_loader.dataset):.4f}")
        tst_loss.append(tst_loss_epoch.detach().cpu().numpy()/len(tst_loader))
        tst_acc.append(tst_acc_epoch.detach().cpu().numpy()/len(tst_loader.dataset))
    
                
    # write your codes here
    print("TRAIN END!")
    trn_loss = np.array(trn_loss)
    trn_acc = np.array(trn_acc)
    tst_loss = np.array(tst_loss)
    tst_acc = np.array(tst_acc)

    trn_loss = [trn_loss, tst_loss]
    trn_acc = [trn_acc, tst_acc]
    
    return trn_loss, trn_acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.to(device)
    model.eval()
    
    tst_loss = 0
    acc = 0
    
    with torch.no_grad():
        for _, (batch) in enumerate(tst_loader):
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)
            
            tst_loss += loss
            acc += torch.sum(torch.argmax(output,dim=1) == labels)
        print(f"TEST FINISHED")
        print(f"TEST LOSS : {tst_loss/len(tst_loader.dataset):.4f}") # reduction mean default
        print(f"TEST ACC : {acc/len(tst_loader.dataset):.4f}")
    # write your codes here

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    ## 0) load configuration
    """ configuration
    
        1) epochs
        2) bs
    
    """
    global EPOCHS
    global BS
    
    EPOCHS = 12
    BS = 32
    
    ## 1) Dataset objects for training and test datasets
    print("DATASET LOAD START!")
    train_dataset = dataset.MNIST('/data/MNIST_assignment/train')
    test_dataset = dataset.MNIST('/data/MNIST_assignment/test')
    
    ## 2) DataLoaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BS, shuffle=False)
    print("DATASET LOAD END!")
    
    ## 3) model
    print("MODEL LOAD START!")
    LeNet = LeNet5()
    LeNet_reg = LeNet5(regularization=True)
    MLP = CustomMLP()
    print("MODEL LOAD END!")
    
    ## 4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
    optimizer_LeNet = torch.optim.SGD([
        {"params" : LeNet.parameters()}],
        lr = 0.01,
        momentum=0.9,)
    
    ## optimizer weight decay : L2 Regularization
    optimizer_LeNet_reg = torch.optim.SGD([
        {"params" : LeNet_reg.parameters()}],
        lr = 0.01,
        momentum=0.9,
        weight_decay=1e-5,)
    
    optimizer_MLP = torch.optim.SGD([
        {"params" : MLP.parameters()}],
        lr = 0.01,
        momentum=0.9,)
    
    
    ## 5) cost function: user torch.nn.CrossEntropyLoss
    criterion_trn = torch.nn.CrossEntropyLoss()
    criterion_tst = torch.nn.CrossEntropyLoss(reduction='sum')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    ## Use test dataset as Validation dataset
    print('LENET TRAIN AND TEST')
    LeNet_trn_loss, LeNet_trn_acc = train(model=LeNet, trn_loader=[train_loader, test_loader], device=device, criterion=criterion_trn, optimizer=optimizer_LeNet)
    LeNet_tst_loss, LeNet_tst_acc = test(model=LeNet, tst_loader=test_loader, device=device, criterion=criterion_tst)
    
    ## Plotting LeNet ##
    plotting(LeNet_trn_loss, LeNet_trn_acc, 'LeNet5')
    
    
    ## Use test dataset as Validation dataset
    print('CUSTOMMLP TRAIN AND TEST')
    MLP_trn_loss, MLP_trn_acc = train(model=MLP, trn_loader=[train_loader, test_loader], device=device, criterion=criterion_trn, optimizer=optimizer_MLP)
    MLP_tst_loss, MLP_tst_acc = test(model=MLP, tst_loader=test_loader, device=device, criterion=criterion_tst)
    
    ## Plotting MLP ##
    plotting(MLP_trn_loss, MLP_trn_acc, 'CustomMLP')
    
    ## Use test dataset as Validation dataset
    print('LENET REG TRAIN AND TEST')
    LeNet_reg_trn_loss, LeNet_reg_trn_acc = train(model=LeNet_reg, trn_loader=[train_loader, test_loader], device=device, criterion=criterion_trn, optimizer=optimizer_LeNet_reg)
    LeNet_reg_tst_loss, LeNet_reg_tst_acc = test(model=LeNet_reg, tst_loader=test_loader, device=device, criterion=criterion_tst)
    
    ## Plotting LeNet ##
    plotting(LeNet_reg_trn_loss, LeNet_reg_trn_acc, 'LeNet5_reg')
    
    
    # write your codes here

if __name__ == '__main__':
    main()
