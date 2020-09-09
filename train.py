import os 
import json
import torch
import argparse 

import models
import dataloaders
from trainer import Trainer
from utils.logger import Logger
from utils.losses import CE_loss, consistency_weight
from models.encoder import Encoder
import inspect 

def main(config, resume):
    torch.manual_seed(42)
    train_logger = Logger()

    # Download data
    root_dir = config['train_supervised']["data_dir"] 
    data_year = config["year"]
    print([eval("dataloaders." + objname) for objname in dir(dataloaders) if type(eval("dataloaders." + objname)) is type])

    dataloaders.Downloader(root_dir, data_year)
    
    # Set number of examples 
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']

    # Dataloaders
    supervised_loader = dataloaders.VOC(config['train_supervised'])
    unsupervised_loader = dataloaders.VOC(config['train_unsupervised'])
    val_loader = dataloaders.VOC(config['val_loader'])

    iters_per_epoch = len(unsupervised_loader)

    # supervised loss 
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    """
    else: sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch,
                                epochs=config['trainer']['epochs'],
                                num_classes = config["num_classes"])
    """

    # Iteration monitoring 
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], 
                                        iters_per_epoch=iters_per_epoch,
                                        rampup_ends=rampup_ends)
    
    # model
    model = models.CCT(num_classes = config['num_classes'],
                        conf=config['model'],
                        sup_loss = sup_loss, 
                        cons_w_unsup = cons_w_unsup,
                        upscale = config['model']['upscale'],
                        num_out_ch = config['model']['num_out_ch'],
                        weakly_loss_w = config['weakly_loss_w'],
                        use_weak_labels = config['use_weak_lables'],

                       ignore_index = val_loader.dataset.ignore_index)                               
    
    print(f'\n{model}\n')


    # Training
    trainer = Trainer(model=model, 
                resume=resume, 
                config=config,
                supervised_loader=supervised_loader,
                unsupervised_loader=unsupervised_loader,
                val_loader=val_loader,
                iters_per_epoch=iters_per_epoch,
                train_logger=train_logger)


    trainer.train()

if __name__=='__main__':

   # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    main(config, args.resume)