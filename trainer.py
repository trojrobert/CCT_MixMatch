import time
import numpy as np
from base import BaseTrainer 

class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iters_per_epoch, 
                val_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, resume, config, iters_per_epoch, train_logger)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))

        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1 

        self.num_classes = config["num_classes"]
        self.mode = self.model.module.mode

        #TRANSFORMS FOR VISUALIZATION
        """
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()
        ])

        self.viz_transform = transforms.Compose([
            transforms.Resize((400,400)),
            transforms.ToTensor()
        ])
        """

        self.start_time = time.time()

    def train(self):

        for epoch in range(self.start_epoch, self.epochs+1):

            results = self._train_epoch(epoch)

            if self.do_validation and epoch % self.config['trainer']['val_per_epoch'] == 0:
                results = self._valid_epoch(epoch)

                self.logger.info('\n\n')

                for k, v in results.items():
                    self.logger.info(f' {str(k):15s}: {v}')

            if self.train_logger is not None: 
                log = {'epoch' : epoch, **results}
                self.train_logger.add_entry(log)

            if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min':self.improved = (log[sefl.mnt_metric] < self.mnt_best)
                    else: self.improved = (log[self.mnt_metric] > self.mnt_best)

                except KeyError:
                    self.logger.warning(f"The metric being tracked ({self.mnt_metric}) has not been calculted. Train stops.")
                    break

                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0

                else:
                    self.not_imporved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stoped')
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)
        self.html_results.save()
