

import torch
import numpy as np
import clip
from torch.utils.data import DataLoader, Dataset

import webdataset as wds
import torch
from PIL import Image
from torchvision import transforms

import torchvision

from prompt import Prompt_classes

from torchvision import datasets, transforms




def test_loader(args, preprocess, device):    
    
    if args.benchmark == 'imagenet':
        imagenet_classes,_ = Prompt_classes('imagenet')
        # imagenet_classes = np.load('./data/ImageNet/imagenet_class_clean.npy')
        imagenet_test = datasets.ImageFolder(root='/data/imagenet_1k/val', transform=preprocess)
        
        if args.ood_dataset == 'iNaturalist':
            ood_dataset = datasets.ImageFolder(root='/data/MOS/iNaturalist', transform=preprocess)
        elif args.ood_dataset == 'SUN':
            ood_dataset = datasets.ImageFolder(root='/data/MOS/SUN', transform=preprocess)
        elif args.ood_dataset == 'Places':
            ood_dataset = datasets.ImageFolder(root='/data/MOS/Places', transform=preprocess)
        elif args.ood_dataset == 'dtd':
            dtd = ood_dataset = datasets.ImageFolder(root='/data/MOS/dtd/images', transform=preprocess)
        
        in_dataloader = DataLoader(imagenet_test, shuffle=False, batch_size = args.bs, num_workers=32)       
        out_dataloader = DataLoader(ood_dataset, shuffle=False, batch_size =args.bs, num_workers=32)
        
    with torch.no_grad():
        texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes]).to(device)
    
    return in_dataloader, out_dataloader, texts_in

def train_loader(args, preprocess, device=None):    
    
    if args.benchmark == 'imagenet':
        if args.multiprocessing_distributed:
            imagenet_classes,_ = Prompt_classes('imagenet')
            imagenet_train = datasets.ImageFolder(root='/data/imagenet_1k/train', transform=preprocess)
            train_sampler = torch.utils.data.distributed.DistributedSampler(imagenet_train,rank=args.rank, num_replicas=args.world_size,shuffle=True)
            in_dataloader = DataLoader(imagenet_train, shuffle=(train_sampler is None), batch_size = args.bs, num_workers=4, sampler=train_sampler)
        else:
            imagenet_classes,_ = Prompt_classes('imagenet')
            imagenet_train = datasets.ImageFolder(root='/data/imagenet_1k/train', transform=preprocess)
            in_dataloader = DataLoader(imagenet_train, shuffle=True, batch_size = args.bs, num_workers=32)    
    elif args.benchmark == 'imagenet100':
        imagenet_classes,_ = Prompt_classes('imagenet')
        imagenet_train = datasets.ImageFolder(root='/data/imagenet_1k/train', transform=preprocess)
        in_dataloader = DataLoader(imagenet_train, shuffle=True, batch_size = args.bs, num_workers=32) 
    elif args.benchmark == 'imagenet10':
        imagenet_classes,_ = obtain_ImageNet10_classes()
        imagenet_train = datasets.ImageFolder(root='/data/MCM_benchmark/ImageNet10', transform=preprocess)
        in_dataloader = DataLoader(imagenet_train, shuffle=True, batch_size = args.bs, num_workers=32) 
        
    with torch.no_grad():
        if args.prompt_name == 'The nice':
            print("Prompt is -The nice-")
            texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes])
        elif args.prompt_name == 'a photo of a':
            print("Prompt is -a photo of a-")
            texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes])
        else:
            print("Prompt is -no-")
            texts_in = clip.tokenize([f"{c}" for c in imagenet_classes])
        
    if args.multiprocessing_distributed:
        return in_dataloader, train_sampler, texts_in
    else:
        return in_dataloader, texts_in



class clip_feature(Dataset):
    def __init__(self, path='/data/MOS/CLIP_im1k_features/train/'):
        super().__init__()
        self.features = torch.load(path+'trainfeature.pt')
        self.targets = torch.load(path+'traintarget.pt')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def _convert_to_rgb(image):
    return image.convert('RGB')

image_preproc = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

def make_loader(url, batch_size=512, num_workers=16, shuffle_size=10000):
    dataset = (
        wds.WebDataset(url)
        .shuffle(shuffle_size)
        .decode("pil")  # 이미지 디코드
        .to_tuple("jpg;png", "txt")  # 이미지와 캡션을 튜플로 묶음
        .map_tuple(image_preproc, lambda x: x)  # 이미지는 전처리, 캡션은 utf-8로 디코드
        .batched(batch_size)
    )

    return torch.utils.data.DataLoader(dataset, num_workers=num_workers)



def obtain_ImageNet10_classes():

    class_dict = {"warplane": "n04552348", "sports car": "n04285008",
                  'brambling bird': 'n01530575', "Siamese cat": 'n02123597',
                  'antelope': 'n02422699', 'swiss mountain dog': 'n02107574',
                  "bull frog": "n01641577", 'garbage truck': "n03417042",
                  "horse": "n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()


def obtain_ImageNet20_classes():

    class_dict = {"n04147183": "sailboat", "n02951358": "canoe", "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
                  "n02917067": "bullet train", "n02317335": "starfish", "n01632458": "spotted salamander", "n01630670": "common newt", "n01631663": "eft",
                  "n02391049": "zebra", "n01693334": "green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",
                  "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[0])}
    return class_dict.values()

def obtain_ImageNet100_classes():
    loc=os.path.join('prompt', 'ImageNet100')
    # sort by values
    with open(os.path.join(loc, 'class_list.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open('./prompt/ImageNet/imagenet_class_index.json') as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set