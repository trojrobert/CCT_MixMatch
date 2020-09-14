import time
import numpy as np
from itertools import cycle
from tqdm import tqdm
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

    def _train_epoch(self, epoch):

        #self.html_results.save()

        self.logger.info('\n')
        self.model.train()

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135)

        else:
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=135)

        #self._reset_metrics()

        for batch_idx in tbar:
            if self.mode == "supervised":
                (input_l, target_l), (input_ul, target_ul) = next(dataloader), (None, None)

            else:
                (input_l, target_l), (input_ul, target_ul) = next(dataloader)
                input_ul, target_ul = input_ul, target_ul

            input_l, target_l = input_l, target_l
            self.optimizer.zero_grad

            total_loss, cur_losses, output = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul, 
                                                        curr_iter= batch_idx, target_ul=target_ul, epoch=epoch-1)
            
            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()

            self._update_losses(cur_losses)
            self._compute_metrics(output, target_l, target_ul, epoch-1)
            logs = self._log_values(cur_losses)

            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                self._write_scalars_tb(logs)

            if batch_idx % int(len(self.unsupervised_loader)*0.9) == 0:
                sel._write_img_tb(input_l, target_l, input_ul, target_ul, outputs, epoch)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            tbar.set_description('T ({}) | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f} |'.format(
                epoch, self.loss_sup.average, self.loss_unsup.average, self.loss_weakly.average,
                self.pair_wise.average, self.mIoU_l, self.mIoU_ul))

            self.lr_scheduler.step(epoch=epoch-1)

        return logs 

    """
    def _reset_metrics(self):

        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()

        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0 
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    """