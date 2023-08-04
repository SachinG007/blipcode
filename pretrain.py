'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
# import ruamel_yaml as yaml
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from data_load import get_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transform.randaugment import RandomAugment
from models.blip_pretrain import blip_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader
from open_clip import create_model_and_transforms, trace_model, get_tokenizer
import numpy as np

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def train(model, data, optimizer, epoch, device, config, output_dir, scheduler):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))    
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    data['train'].set_epoch(epoch)
    data_loader = data['train'].dataloader
    print(f"Num Batches : {data_loader.num_batches}")

    for i, (image, caption) in enumerate(data_loader):
        
        if epoch==0:
            warmup_lr_schedule(optimizer, i, 3000, 1e-6, 3e-4)
            
        optimizer.zero_grad()
        
        image = image.to(device,non_blocking=True)
        
        # ramp up alpha in the first 2 epochs
        alpha = config['alpha']*min(1,(epoch*data_loader.num_batches+i)/(2*data_loader.num_batches)) 

        loss_ita, loss_itm, loss_lm = model(image, caption, alpha = alpha)  
        if i%1000==0:
            print(f"Step : {i}, Loss_ita : {loss_ita}, Loss_itm : {loss_itm} Loss_lm : {loss_lm}")
        if i%1000==0:
            print(f"Checkpoint Saved at Step : {i}")
            model_without_ddp = model.module
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint.pth'))  
        loss = loss_ita + loss_itm + loss_lm  

        loss.backward()
        optimizer.step()
        step = data_loader.num_batches*epoch+i
        scheduler(step)

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    # datasets = [create_dataset('pretrain', config, min_scale=0.2)]
    # print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    # samplers = create_sampler(datasets, [True], num_tasks, global_rank)         

    # data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]     
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    preprocess_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(0.2, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    preprocess_val = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    data = get_data(args.train_data, args.batch_size, (preprocess_train, preprocess_val), epoch=0, tokenizer=get_tokenizer("ViT-B-32"), valid_file=args.valid_file, train_samples=args.train_samples)
    # data_loader = data["train"].dataloader
    #### Model #### 
    print("Creating model")
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                            vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])

    model = model.to(device)   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0003, weight_decay=config['weight_decay'])
    warmup_steps = 10000
    total_steps = data["train"].dataloader.num_batches
    scheduler = cosine_lr(optimizer, 0.0003, warmup_steps, total_steps)
    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
        
    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        
        train_stats = train(model, data, optimizer, epoch, device, config, args.output_dir, scheduler) 
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()        
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='output/pretrain')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--valid_file', default=None)
    parser.add_argument('--batch_size', default=75, type=int)
    parser.add_argument('--train_samples', default=0, type=int)

    args = parser.parse_args()
    print(args.config)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)