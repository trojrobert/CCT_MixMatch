import math, datetime, os, json
import logging
import torch
from utils import logger, helpers
import utils.lr_scheduler


from torch.utils import tensorboard

class BaseTrainer:

    def __init__(self, model, resume, config, iters_per_epoch, train_logger=None):
        self.model = model
        self.config = config

        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.improved = False 

        # use available resources 
        self.device, available_gpus = self._get_available_devices(self.config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        self.model.to(self.device)

        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        trainable_params = [{'params': filter(lambda p: p.requires_grad, self.model.module.get_other_params())},
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_backbone_params()), 
                    'lr': config['optimizer']['args']['lr'] / 10}]

        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        model_params = sum([i.shape.numel() for i in list(model.parameters())])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])

        assert opt_params == model_params, 'some params are missing in the opt'

        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler'])(optimizer=self.optimizer, num_epochs=self.epochs,
                                    iters_per_epoch=iters_per_epoch)

        self.monitor = cfg_trainer.get('monitor', 'off')

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0

        else: 
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        run_name = config['experim_name']
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], run_name)

        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')

        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(cfg_trainer['log_dir'], run_name)
        self.writer = tensorboard.SummaryWriter(writer_dir)
        #self.html_results = HTML(web_dir=config['trainer']['save_dir'], exp_name=config['experim_name'],
        #                    save_name=config['experim_name'], config=config, resume=resume)
        
        if resume:self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        """Check for resources available"""
        sys_gpu = torch.cuda.device_count()

        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0

        elif n_gpu > sys_gpu:
            self.logger.warnin(f'No of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))

        return  device, available_gpus

def get_instance(module, name, config, *args):

    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
    