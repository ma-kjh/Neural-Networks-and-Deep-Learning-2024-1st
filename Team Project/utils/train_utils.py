
import torch
import wandb
from tqdm import tqdm
import os
import clip

from nltk.corpus import wordnet
from prompt import Prompt_classes
import numpy as np


def train(args, model, in_dataloader, in_texts, device):
    save_acc = 0
    print('dispersion')
    p_name = args.prompt_name.replace(' ','_')
    wandb.init(project=f'NeurIPS2024',entity="daintlab",name=f'{args.methods}_{args.models}_{args.benchmark}_{args.lr}_{args.bs}_epoch_{args.epochs}_{args.clip}_lam1_{args.lam1}_lam2_{args.lam2}_seed_{args.seed}_{p_name}_singleprecision')
    print(args)
    
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    
    words = [word.lower() for word in wordnet.words() if wordnet.synsets(word, pos='n')]
    imagenet_classes,_ = Prompt_classes('imagenet')
    words = np.setdiff1d(words, imagenet_classes)
    random_title = clip.tokenize([f"a photo of a {c}" for c in words])
    
    devices = list(range(torch.cuda.device_count()))
    print(devices)
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)

    optimizer = torch.optim.AdamW([
        {"params" : model.parameters()}],
        lr=args.lr,
        weight_decay=0.1) 
    
    from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
    num_batches = 1281167//args.bs+1
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.epochs*num_batches, cycle_mult=1.0, max_lr=args.lr, min_lr=0, warmup_steps=500, gamma=1.0)
    
    EPOCH = args.epochs
    
    for epoch in range(1,EPOCH+1):
        model.train()
        
        for i, (batch) in enumerate(tqdm(in_dataloader)):
            wandb.log({f"Train learning rate": scheduler.get_lr()[0]})
            optimizer.zero_grad()
            ground_truth_text = torch.arange(batch[0].shape[0],dtype=torch.long,device=device)
            
            images, labels = batch                         
            
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            images = images.to(device)
            texts = in_texts[labels]

            
            image_embeddings, text_embeddings, scale = model(images, texts)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            norm_text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

            logits_per_image = (scale[0] * image_embeddings @ norm_text_embeddings.T)
            logits_per_text = logits_per_image.T
            
            # logits_per_image_text_random = (scale[0] * image_embeddings @ random_text_embeddings.T)
            # logits_per_text_random = logits_per_image_text_random.T
                
            image_loss = loss_img(logits_per_image, ground_truth_text)
            text_loss = loss_txt(logits_per_text, ground_truth_text)
            
            t_loss_data_text = -torch.logsumexp(logits_per_text,dim=1)
            # t_loss_sample_text = -torch.logsumexp(logits_per_text_random,dim=1)
            
            t_loss_data_image = -torch.logsumexp(logits_per_image,dim=1)
                # t_loss_data_image = -torch.logsumexp(logits_per_image_random,dim=1)
                
            total_loss = (text_loss + args.lam1 * t_loss_data_text.mean())/2 + (image_loss + args.lam1 * t_loss_data_image.mean())/2
            
            wandb.log({f"Train Text Loss": text_loss})
            wandb.log({f"Train Image Loss": image_loss})
            wandb.log({f"Train total Loss": total_loss})

            # wandb.log({f"Train Text sample Loss": t_loss_sample_text.mean()})
            # wandb.log({f"Train Text data Loss": t_loss_data_text.mean()})

            total_loss.backward()
            optimizer.step()
            scheduler.step()
    
        save_name = f"./checkpoints/{args.benchmark}/{args.methods}_{args.clip}_{args.bs}_{args.lr}_lam1_{args.lam1}_lam2_{args.lam2}_seed_{args.seed}_{args.prompt_name.replace(' ','_')}_singleprecision_epoch_{args.epochs}_norandom_dispersion"
            
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_name}/model_{args.methods}_{epoch}.pt")
