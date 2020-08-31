import os 
import json
import torch
import argparse 

import dataloaders

import inspect 

def main(config, resume):

    #Download data
    root_dir = config['train_supervised']["data_dir"] 
    data_year = config["year"]
    print([eval("dataloaders." + objname) for objname in dir(dataloaders) if type(eval("dataloaders." + objname)) is type])

    dataloaders.Downloader(root_dir, data_year)
    
    # DATA LOADERS
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']

    supervised_loader = dataloaders.VOC(config['train_supervised'])
    unsupervised_loader = dataloaders.VOC(config['train_unsupervised'])
    val_loader = dataloaders.VOC(config['val_loader'])

    iter_per_epoch = len(unsupervised_loader)
 

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