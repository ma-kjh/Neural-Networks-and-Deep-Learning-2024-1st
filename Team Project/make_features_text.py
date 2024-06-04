import numpy as np
import clip
from tqdm import tqdm
import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import  MNIST, CIFAR10, SVHN, CIFAR100, ImageFolder
import torchvision.transforms as transforms
import torchvision

from prompt import Prompt_classes

import argparse

from torch.autograd import Variable

from utils.common import setup_seed
from utils.metrics import get_measures
from utils.load_model import set_model_clip
from utils.loader_list import test_loader_list

import nltk
from nltk.corpus import wordnet as wn
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description = 'Evaluates CLIP Out-of-distribution Detection',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--models', type=str, default='ViT-B/16')
parser.add_argument('--clip', type=str, default='openai')
parser.add_argument('--ckpt', type=str, default='./')
parser.add_argument('--ood-dataset',type=str, default='iNaturalist')
parser.add_argument('--methods', type=str, default='flyp')
parser.add_argument('--benchmark',type=str, default='imagenet')
parser.add_argument('--prompt',type=str, default='The nice')
parser.add_argument('--dir', type=str, default='./features/zero-shot-The-nice/')
parser.add_argument('--bs', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sim', type=int, default=1.0)
parser.add_argument('--is-train', default=False,action="store_true")


parser.add_argument('--multiprocessing-distributed', default=False,action="store_true")

args = parser.parse_args()

print(args.ckpt)
## feature 저장할 곳  ##

# dirname = f'./features/zero-shot-a-photo-of/'
# dirname = f'./features/zero-shot-The-nice/'
# dirname = f'./features/no-prompt/'
# dirname = f'./features/flyp-1e-06-a-photo-of/'
# dirname = f'./features/flyp-1e-06-The-nice/'
# dirname = f'./features/energy-1e-06-text+image-The-nice/'
# dirname = f'./features/energy-1e-06-text+image-a-photo-of/'
# dirname = f'./features/energy-1e-06-0.1-0.000-text+image-The-nice/'
# dirname = f'./features/energy-1e-05-0.001-0.000-text+image-a-photo-of-a'
# dirname = f'./features/flyp-1e-05-The-nice/'
# dirname = f'./features/{args.ckpt}-The-nice'
# dirname = f'./features/{args.ckpt}-no-prompt'

# dirname = './features/test'
dirname = args.dir

os.makedirs(f'{dirname}', exist_ok=True)
######################


## 모델 불러오기 ##
print('model load !')
model, _, preprocess = set_model_clip(args)
in_dataloader, out_dataloader, texts_in = test_loader_list(args, preprocess, device)
imagenet_classes,_ = Prompt_classes('imagenet')
if args.prompt == 'The nice':
    print('Prompt -The nice-')
    texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes]).to(device)
elif args.prompt == 'a photo of a':
    print('Prompt -A photo of a-')
    texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes]).to(device)
elif args.prompt == 'a good photo of a':
    print('Prompt -A photo of a-')
    texts_in = clip.tokenize([f"a good photo of a {c}" for c in imagenet_classes]).to(device)
elif args.prompt == 'no':
    print('Prompt -no-')
    texts_in = clip.tokenize([f"{c}" for c in imagenet_classes]).to(device)
# texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes]).to(device)
# texts_in = clip.tokenize([f"{c}" for c in imagenet_classes]).to(device)
model.to(device)
model = model.eval()
print('model load finished !')
################

# #### 
# def extract_words(pos):
#     words = set()
#     for synset in list(wn.all_synsets(pos=pos)):
#         for lemma in synset.lemmas():
#             words.add(lemma.name())
#     return words

# # Extracting all unique nouns
# nouns = extract_words('n')
# print(f"Found {len(nouns)} unique nouns.")

# # Extracting all unique adjectives
# adjectives = extract_words('a')  # 'a' for adjectives
# print(f"Found {len(adjectives)} unique adjectives.")
# ####

# ## wordnet와서 토큰화 ##
# print('wordnet tokenize !')
# words_noun = [word.replace('_',' ') for word in nouns]
# words_adj = [word for word in adjectives]

# # wordnet_texts_noun = clip.tokenize([f"a photo of a {c}" for c in words_noun]).cuda()
# wordnet_texts_noun = clip.tokenize([f"a photo of a {c}" for c in words_noun]).cuda()
# # wordnet_texts_noun = clip.tokenize([f"{c}" for c in words_noun]).cuda()
# wordnet_texts_adj = clip.tokenize([f"This is a {c} photo" for c in words_adj]).cuda()
# # wordnet_texts_adj = clip.tokenize([f"{c}" for c in words_adj]).cuda()
# print('wordnet tokenize finished !')
# ####################

# ## wordnet token -> wordnet embedding ##
# print('wordnet token -> wordnet embedding !')
# encoded_texts_noun = []
# with torch.no_grad():
#     for i in range(0, len(wordnet_texts_noun), 1000):
#         torch.cuda.empty_cache()
#         if args.ckpt == './':
#             a = model.encode_text(wordnet_texts_noun[i:i+1000])    
#         else:
#             a = model.module.encode_text(wordnet_texts_noun[i:i+1000])
#         a_cpu = a.cpu()
#         encoded_texts_noun.append(a_cpu)
        
# encoded_texts_adj = []
# with torch.no_grad():
#     for i in range(0, len(wordnet_texts_adj), 1000):
#         torch.cuda.empty_cache()
#         if args.ckpt == './':
#             a = model.encode_text(wordnet_texts_adj[i:i+1000])    
#         else:
#             a = model.module.encode_text(wordnet_texts_adj[i:i+1000])
#         a_cpu = a.cpu()
#         encoded_texts_adj.append(a_cpu)
        
# print('wordnet token -> wordnet embedding finished!')
# ########################################

# print('wordnet tokenize and embedding !')
# #####################################################


#####################################################
directory_path = './prompt/Neglabel/txtfiles/'

file_list = os.listdir(directory_path)

encoded_texts_noun = []
encoded_texts_adj = []
no_encoded_texts_noun = []
no_encoded_texts_adj = []
tt_noun = []
tt_adj = []
for filename in file_list:
# 파일을 열고 각 줄을 리스트로 읽기
    if 'adj' in filename:
        with open(directory_path + filename, 'r') as file:
            neg = [line.strip() for line in file]
        neg_text = clip.tokenize([f"This is a {c} photo" for c in neg]).cuda()
        tt_adj = np.array([f"This is a {c} photo" for c in neg])
        with torch.no_grad():
            for i in range(0, len(neg_text), 1000):
                print(i)
                torch.cuda.empty_cache()
                if args.ckpt == './':
                    a = model.module.encode_text(neg_text[i:i+1000])
                else:
                    a = model.module.encode_text(neg_text[i:i+1000])
                # GPU에서 계산된 결과를 CPU로 옮기기
                a_cpu = a.cpu()
                # 결과 리스트에 추가
                encoded_texts_adj.append(a_cpu)
                no_encoded_texts_adj.append(tt_adj[i:i+1000])
    else:
        with open(directory_path + filename, 'r') as file:
            neg = [line.strip() for line in file]
        if args.prompt == 'The nice':
            neg_text = clip.tokenize([f"The nice {c}" for c in neg]).cuda()
            tt_noun = np.array([f"The nice {c}" for c in neg])
        elif args.prompt == 'a photo of a':
            neg_text = clip.tokenize([f"a photo of a {c}" for c in neg]).cuda()
            tt_noun = np.array([f"a photo of a {c}" for c in neg])
        elif args.prompt == 'a good photo of a':
            neg_text = clip.tokenize([f"a good photo of a {c}" for c in neg]).cuda()
            tt_noun = np.array([f"a photo of a {c}" for c in neg])
        elif args.prompt == 'no':
            neg_text = clip.tokenize([f"{c}" for c in neg]).cuda()
            tt_noun = np.array([f"{c}" for c in neg])
        with torch.no_grad():
            for i in range(0, len(neg_text), 1000):
                print(i)
                torch.cuda.empty_cache()
                if args.ckpt == './':
                    a = model.module.encode_text(neg_text[i:i+1000])
                else:
                    a = model.module.encode_text(neg_text[i:i+1000])
                # GPU에서 계산된 결과를 CPU로 옮기기
                a_cpu = a.cpu()
                # 결과 리스트에 추가
                encoded_texts_noun.append(a_cpu)
                no_encoded_texts_noun.append(tt_noun[i:i+1000])
print('wordnet tokenize and embedding finished !')
###################################################


# ###########################################################
## imagenet text token -> imagenet text embedding ##
## texts_in -> test_loader_list 에서 가져옴 ###########
print('imagenet token -> imagenet textembedding !')
with torch.no_grad():
    if args.ckpt == './':
        imagenet_texts = model.module.encode_text(texts_in)    
    else:
        imagenet_texts = model.module.encode_text(texts_in)
    imagenet_texts_cpu = imagenet_texts.cpu()
    
imagenet_texts_cpu_norm = imagenet_texts_cpu / imagenet_texts_cpu.norm(dim=-1,keepdim=True)
torch.save(imagenet_texts_cpu_norm,  dirname + 'imagenet_texts_norm.pt')
print('imagenet token -> imagenet textembedding finished!')
# ####################################################

########################################################
### NegMining ###
print('NegMining !')
wordnet_noun = torch.cat(encoded_texts_noun)
wordnet_adj = torch.cat(encoded_texts_adj)

wordnet_noun_cos = wordnet_noun @ imagenet_texts_cpu.T
wordnet_adj_cos = wordnet_adj @ imagenet_texts_cpu.T

noun_q_005 = torch.quantile(wordnet_noun_cos, 0.95, dim=-1)
adj_q_005 = torch.quantile(wordnet_adj_cos, 0.95, dim=-1)
_, noun_idx= torch.topk(noun_q_005,k=8500, largest=False)
_, adj_idx= torch.topk(adj_q_005,k=1500, largest=False)

neg_label_noun = wordnet_noun[noun_idx]
neg_label_adj = wordnet_adj[adj_idx]

neg_label_noun_norm = neg_label_noun / neg_label_noun.norm(dim=-1, keepdim=True)
neg_label_adj_norm = neg_label_adj / neg_label_adj.norm(dim=-1, keepdim=True)

torch.save(neg_label_noun_norm,  dirname + 'neg_label_noun_norm_8500.pt')
torch.save(neg_label_adj_norm,  dirname + 'neg_label_adj_norm_1500.pt')
print('NegMining finished!')
############
# import pdb; pdb.set_trace()
no_encoded_texts_adj_list = []
no_encoded_texts_noun_list = []
for arr in no_encoded_texts_adj:
    for i in arr:
        no_encoded_texts_adj_list.append(i)
for arr in no_encoded_texts_noun:
    for i in arr:
        no_encoded_texts_noun_list.append(i)

# import pdb; pdb.set_trace()
######################################################
# t = np.array(no_encoded_texts_adj_list)
# tt = np.array(no_encoded_texts_noun_list)
# neg_adj = np.load('./features/zero-shot-The-nice/neg_label_adj_real_1500.npy')
# neg_noun = np.load('./features/zero-shot-The-nice/neg_label_noun_real_8500.npy')
# np.setdiff1d(words, imagenet_classes)

# np.save(np.array(no_encoded_texts_adj_list)[adj_idx] ,'./features/zero-shot-a-photo-of-a/neg_label_adj_norm_8500_real.npy')
# np.save(np.array(no_encoded_texts_noun_list)[noun_idx], './features/zero-shot-a-photo-of-a/neg_label_noun_norm_8500_real.npy')
if args.prompt == 'a photo of a':
    np.save('./features/zero-shot-a-photo-of-a/neg_label_adj_1500_real.npy', np.array(no_encoded_texts_adj_list)[adj_idx] )
    np.save('./features/zero-shot-a-photo-of-a/neg_label_noun_8500_real.npy',np.array(no_encoded_texts_noun_list)[noun_idx])
elif args.prompt == 'The nice':
    np.save('./features/zero-shot-The-nice/neg_label_adj_1500_real.npy', np.array(no_encoded_texts_adj_list)[adj_idx] )
    np.save('./features/zero-shot-The-nice/neg_label_noun_8500_real.npy',np.array(no_encoded_texts_noun_list)[noun_idx])
######################################################


# #################################################################
filename = './prompt/Neglabel/selected_neg_labels_in1k_10k.txt'

# 파일을 열고 각 줄을 리스트로 읽기


with open(filename, 'r') as file:
    neg = [line.strip() for line in file]
    
neg_text = clip.tokenize(neg).cuda()

neg_encoded_texts = []
with torch.no_grad():
    for i in range(0, len(neg_text), 1000):
        print(i)
        torch.cuda.empty_cache()
        if args.ckpt == './':
            a = model.module.encode_text(neg_text[i:i+1000])
        else:
            a = model.module.encode_text(neg_text[i:i+1000])
        # GPU에서 계산된 결과를 CPU로 옮기기
        a_cpu = a.cpu()
        # 결과 리스트에 추가
        neg_encoded_texts.append(a_cpu)
neg_neg = torch.cat(neg_encoded_texts) / torch.cat(neg_encoded_texts).norm(dim=-1,keepdim=True)
torch.save(neg_neg, dirname + 'Neg_github_texts_norm.pt')
##################################################################

# 파일을 열고 각 줄을 리스트로 읽기



########### neg_zero-shot #############################
if args.prompt == 'a photo of a':
    neg_adj = np.load('./features/zero-shot-a-photo-of-a/neg_label_adj_1500_real.npy')
    neg_noun = np.load('./features/zero-shot-a-photo-of-a/neg_label_noun_8500_real.npy')
elif args.prompt == 'The nice':
    neg_adj = np.load('./features/zero-shot-The-nice/neg_label_adj_1500_real.npy')
    neg_noun = np.load('./features/zero-shot-The-nice/neg_label_noun_8500_real.npy')
elif args.prompt == 'no':
    neg_adj = np.load('./features/zero-shot-no/neg_label_adj_1500_real.npy')
    neg_noun = np.load('./features/zero-shot-no/neg_label_noun_8500_real.npy')
elif args.prompt == 'a good photo of a':
    neg_adj = np.load('./features/zero-shot-a-good-photo-of-a/neg_label_adj_1500_real.npy')
    neg_noun = np.load('./features/zero-shot-a-good-photo-of-a/neg_label_noun_8500_real.npy')
    
neg_adj = clip.tokenize(neg_adj)
neg_noun = clip.tokenize(neg_noun)

neg_text = torch.cat([neg_adj, neg_noun]).cuda()
    
# neg_text = clip.tokenize(neg).cuda()

neg_encoded_texts = []
with torch.no_grad():
    for i in range(0, len(neg_text), 1000):
        print(i)
        torch.cuda.empty_cache()
        if args.ckpt == './':
            a = model.module.encode_text(neg_text[i:i+1000])
        else:
            a = model.module.encode_text(neg_text[i:i+1000])
        # GPU에서 계산된 결과를 CPU로 옮기기
        a_cpu = a.cpu()
        # 결과 리스트에 추가
        neg_encoded_texts.append(a_cpu)
neg_neg = torch.cat(neg_encoded_texts) / torch.cat(neg_encoded_texts).norm(dim=-1,keepdim=True)
torch.save(neg_neg, dirname + 'Neg_zero_shot_texts_norm.pt')
print("NEG ZERO !")
#########################################################
# import pdb; pdb.set_trace()

# ##############################################################
# # 모델에서 image feature 저장 ##
# print('image features !')
# encoded_images = []
# tqdm_object = tqdm(in_dataloader, total=len(in_dataloader))
# with torch.no_grad():
#     for batch_idx, (images, labels) in enumerate(tqdm_object):
#         bz = images.size(0)
#         images = images.cuda()
#         if args.ckpt == './':
#             image_embeddings = model.module.encode_image(images)    
#         else:
#             image_embeddings = model.module.encode_image(images)
#         image_embeddings_cpu = image_embeddings.cpu()
#         encoded_images.append(image_embeddings_cpu)

# Imagenet_images = torch.cat(encoded_images)
# Imagenet_images_norm = Imagenet_images / Imagenet_images.norm(dim=-1,keepdim=True)
# torch.save(Imagenet_images_norm,  dirname + 'imagenet_images_norm.pt')
# print('image features finished!')
# # #################################################################


# # ################################################################
# print('ood features !')
# ood_names = ["iNaturalist","SUN", "Places", "Textures"]

# for i, (name, loader) in enumerate(zip(ood_names,out_dataloader)):
#     print(name)
#     tqdm_object = tqdm(loader, total=len(loader))
#     ood_images_feature=[]
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(tqdm_object):
#             bz = images.size(0)
#             images = images.cuda()
#             if args.ckpt == './':
#                 image_embeddings = model.module.encode_image(images)    
#             else:
#                 image_embeddings = model.module.encode_image(images)
#             image_embeddings_cpu = image_embeddings.cpu()
#             if name == 'iNaturalist':
#                 ood_images_feature.append(image_embeddings_cpu)
#             if name == 'SUN':
#                 ood_images_feature.append(image_embeddings_cpu)
#             if name == 'Places':
#                 ood_images_feature.append(image_embeddings_cpu)
#             if name == 'Textures':
#                 ood_images_feature.append(image_embeddings_cpu)
#         ood_images_feature = torch.cat(ood_images_feature)
#         ood_images_feature /= ood_images_feature.norm(dim=-1,keepdim=True)
#         if i == 0:
#             torch.save(ood_images_feature, dirname + 'inaturalist_images_norm.pt')
#         if i == 1:
#             torch.save(ood_images_feature, dirname + 'sun_images_norm.pt')
#         if i == 2:
#             torch.save(ood_images_feature, dirname + 'places_images_norm.pt')
#         if i == 3:
#             torch.save(ood_images_feature, dirname + 'textures_images_norm.pt')
# # ################################################################################