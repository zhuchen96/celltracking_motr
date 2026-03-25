# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import time
from pathlib import Path
import numpy as np
import sacred
import torch
import yaml
from torch.utils.data import DataLoader

import trackformer.util.misc as utils
from trackformer.datasets import build_dataset
from trackformer.engine import evaluate, train_one_epoch
from trackformer.models import build_model

filepath = Path(__file__)
yaml_file_paths = (filepath.parents[1] / 'cfgs').glob("*.yaml")
yaml_files = [yaml_file.stem.split('train_')[1] for yaml_file in yaml_file_paths]

ex = sacred.Experiment('train')

def train(respath, dataset) -> None:
    
    # if (respath / 'checkpoint.pth').exists():
    #     ex.add_config(str(respath / 'config.yaml'))
    # else:
    
    ex.add_config(str(filepath.parents[1] / 'cfgs' / ('train_' + dataset + '.yaml')))

    config = ex.run_commandline().config
    args = utils.nested_dict_to_namespace(config)

    # if (respath / 'config.yaml').exists():
    #     args.resume = str(respath / 'checkpoint.pth')

    Path(args.output_dir).mkdir(exist_ok=True)

    if args.dn_track or args.dn_object:
        assert args.use_dab, f'DAB-DETR is needed to use denoised boxes for tracking / object detection. args.use_dab is currently set to {args.use_dab}'

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.CoMOT:
        assert not args.share_bbox_layers, 'Are you sure you want to share layers when using CoMOT'

    if args.num_OD_layers:
        assert args.num_OD_layers == 1

    if args.debug:
        args.num_workers = 0

    if args.output_dir:
        args.output_dir = str(args.output_dir)
        yaml.dump(
            vars(args),
            open(Path(args.output_dir) / 'config.yaml', 'w'), allow_unicode=True)

    args.output_dir = Path(args.output_dir)
    print(args)

    val_output_folder = 'val_outputs'
    train_output_folder = 'train_outputs'
    utils.create_folders(train_output_folder,val_output_folder,args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    seed = 248
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('NUM TRAINABLE MODEL PARAMS:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names + args.lr_linear_proj_names + ['layers_track_attention']) and p.requires_grad],
         "lr": args.lr,},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
         "lr":  args.lr * args.lr_linear_proj_mult}]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    args_drop_num = args.epochs // args.lr_drop
    lr_drop = [args.lr_drop * i for i in range(1,args_drop_num+1)] 

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_drop, gamma=0.1)

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker)

    data_loader_val = DataLoader(
        dataset_val, args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker)

    if args.resume:
        model_without_ddp = utils.load_model(model_without_ddp,args,param_dicts,optimizer,lr_scheduler)

    if args.eval_only:
        raise NotImplementedError
        evaluate(model, criterion, data_loader_val, device, args.output_dir, args, 0)
        return
    
    assert args.start_epoch < args.epochs + 1
    
    print("Start training")
    model.train_model = True # detr_tracking script will process multiple sequential frames

    for epoch in range(args.start_epoch, args.epochs + 1):

        start_epoch = time.time() # Measure time for each epoch
        
        # set epoch for reproducibliity
        dataset_train.set_epoch(epoch)
        dataset_val.set_epoch(epoch) 

        # TRAIN
        train_metrics = train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, args)

        lr_scheduler.step()

        # VAL
        val_metrics = evaluate(model, criterion, data_loader_val,args, epoch)

        # Save loss and metrics in a pickle file
        utils.save_metrics_pkl(train_metrics,args.output_dir,'train',epoch=epoch)  
        utils.save_metrics_pkl(val_metrics,args.output_dir,'val',epoch=epoch)  

        # plot loss
        utils.plot_loss_and_metrics(args.output_dir)

        checkpoint_paths = [args.output_dir / 'checkpoint.pth']

        if args.output_dir:
            if args.save_model_interval and not epoch % args.save_model_interval:
                checkpoint_paths.append(str(args.output_dir / f"checkpoint_epoch_{epoch}.pth"))

            for checkpoint_path in checkpoint_paths:
                args.output_dir = str(args.output_dir)
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, str(checkpoint_path))
                args.output_dir = Path(args.output_dir)

        # Display and save time epoch took
        total_epoch_time = time.time() - start_epoch
        print(f'Epoch took {str(datetime.timedelta(seconds=int(total_epoch_time)))}')
        with open(str(args.output_dir / "training_time.txt"), "a") as f:
            f.write(f"Epoch {epoch}: {str(datetime.timedelta(seconds=int(total_epoch_time)))}\n")

    # Display and save total time training took
    total_time = utils.get_total_time(args)
    print('Training time {}'.format(total_time))

@ex.config
def my_config():
    dataset = yaml_files[0]  # Default dataset

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

if __name__ == '__main__':
    # Parse the dataset from the command line
    args = ex.run_commandline().config
    dataset = 'sim'

    respath = filepath.parents[1] / 'results' / dataset

    respath.mkdir(exist_ok=True)

    train(respath, dataset)
