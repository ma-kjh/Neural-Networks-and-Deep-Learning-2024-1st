

import torch
import clip
from torch.utils.data import DataLoader


import torchvision

from prompt import Prompt_classes

from torchvision import datasets, transforms


def test_loader_list(args, preprocess, device):    
    
    if args.benchmark == 'imagenet':
        imagenet_classes,_ = Prompt_classes('imagenet')
        imagenet_test = datasets.ImageFolder(root='/data/imagenet/val', transform=preprocess)
        
        
        iNaturalist = datasets.ImageFolder(root='/data/MOS/iNaturalist', transform=preprocess)
        SUN = datasets.ImageFolder(root='/data/MOS/SUN', transform=preprocess)
        Places = datasets.ImageFolder(root='/data/MOS/Places', transform=preprocess)
        dtd = datasets.ImageFolder(root='/data/MOS/dtd/images', transform=preprocess)
        
        in_dataloader = DataLoader(imagenet_test, shuffle=False, batch_size = args.bs, num_workers=16)
        
        iNaturalist = DataLoader(iNaturalist, shuffle=False, batch_size= args.bs, num_workers=16)
        SUN = DataLoader(SUN, shuffle=False, batch_size= args.bs, num_workers=16)
        Places = DataLoader(Places, shuffle=False, batch_size= args.bs, num_workers=16)
        dtd = DataLoader(dtd, shuffle=False, batch_size= args.bs, num_workers=16)
        
        out_dataloader = [iNaturalist, SUN, Places, dtd]
    
    with torch.no_grad():
        # texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes]).to(device)
        # texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes]).to(device)
        texts_in = clip.tokenize([f"{c}" for c in imagenet_classes]).to(device)
    
    return in_dataloader, out_dataloader, texts_in
