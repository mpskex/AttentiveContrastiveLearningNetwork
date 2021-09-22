import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np

import ttach as tta

from fgvc_cl.utils.AverageMeter import accuracy, AverageMeter
from fgvc_cl.utils.Register import REGISTER_TRAINER, REGISTRY, print_loaded
from fgvc_cl.utils.Checkpoints import save_checkpoint, load_checkpoint
from fgvc_cl.data.ImageIndexDataset import ImageIndexDataset
from fgvc_cl.utils.WarmUpLR import GradualWarmupScheduler

tta_transforms = tta.aliases.five_crop_transform(448, 448)

@REGISTER_TRAINER('APINetTrainer')
class APINetTrainer:
    """
    API Net Trainer
    """
    def __init__(self, param, local_rank=0):
        self.param = param
        if param.name is not None:
            self.name = param.name
        else:
            self.name = '_'.join([param.model, param.backbone, str(time.time()).split('.')[0]])
            self.param.name = self.name
        if local_rank == 0:
            print("== CONFIGURATION ==")
            print(str(self.param))
        self.local_rank = local_rank
        print("rank %d initialized!"%self.local_rank)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = np.random.randint(low=0, high=2**32)
        self.best_prec = 0
        self.train_set = ImageIndexDataset(REGISTRY['DATALOADER'][param.data_loader](self.param, split='train'))
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=param.batch_size,
                                                        sampler=self.train_sampler,
                                                        num_workers=4, pin_memory=True)
        self.param.class_num = self.train_set.class_num

        self.valid_set = ImageIndexDataset(REGISTRY['DATALOADER'][param.data_loader](self.param, split='valid'))
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_set)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=param.batch_size,
                                                        sampler=self.valid_sampler,
                                                        num_workers=4, pin_memory=True)
        self.valid_tta = False

        self.model = REGISTRY['MODEL'][param.model](self.param, local_rank=self.local_rank)
        self.lr_schedulers = [GradualWarmupScheduler(optimizer, 1.0, 20,
                                                     after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                                                         optimizer,
                                                         self.param.epochs)) for optimizer in self.model.optimizers]

    def sync_metrics(self, *args):
        for n in args:
            torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        return list(map(lambda x: x / world_size, args))


    def train(self):
        step = 0
        # this zero gradient update is needed to avoid a warning message, issue #8.
        list(map(lambda x: x.zero_grad(), self.model.optimizers))
        list(map(lambda x: x.step(), self.model.optimizers))

        for epoch in range(self.param.epochs):
            #   Learning rate update
            list(map(lambda x: x.step(), self.lr_schedulers))

            end = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            self.train_sampler.set_epoch(epoch)
            for i, (input, target, index) in enumerate(self.train_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                input_var = input.to(self.device)
                target_var = target.to(self.device)

                loss, logits, targets = self.model.train(input_var, target_var,
                                                         epoch=epoch,
                                                         _iter=step, 
                                                         image_ids=index, 
                                                         dataset=self.train_set)

                # measure accuracy and record loss
                loss = torch.as_tensor(loss).to(device=self.device)
                prec1 = accuracy(logits, targets, 1)
                prec5 = accuracy(logits, targets, 5)

                result = self.sync_metrics(loss, prec1, prec5)

                loss, prec1, prec5 = list(map(lambda x: x.item(), result))

                losses.update(loss, 2 * self.param.batch_size)
                top1.update(prec1, 4 * self.param.batch_size)
                top5.update(prec5, 4 * self.param.batch_size)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 20 == 0 and self.local_rank == 0:
                    print('Time: {time}\nStep: {step}\t Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        top1=top1, top5=top5, step=step, time=time.asctime(time.localtime(time.time()))))

                if i == len(self.train_loader) - 1:
                    prec1 = self.validate()

                    if self.local_rank == 0:
                        # remember best prec@1 and save checkpoint
                        is_best = prec1 >= self.best_prec
                        self.best_prec = max(prec1, self.best_prec)
                        if self.local_rank == 0:
                            save_checkpoint(
                                self.name,
                                self.param,
                                'checkpoint-%s.pth.tar' % (epoch+1),
                                {
                                    'epoch': epoch + 1,
                                    'best_prec': prec1,
                                    'state_dict': self.model.state_dict()
                                },
                                is_best)
                step = step + 1
        if self.local_rank == 0:
            print("Final Top1 is %.4f"%self.best_prec)
            prec1 = self.validate(with_tta=True)
        return step

    def validate(self, with_tta=False):
        batch_time = AverageMeter()
        softmax_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        end = time.time()
        if self.valid_tta != with_tta:
            if with_tta:
                self.valid_set = ImageIndexDataset(REGISTRY['DATALOADER'][self.param.data_loader](self.param,
                                                                                                  split='valid',
                                                                                                  tta=with_tta))
                self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_set)
                self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.param.batch_size,
                                                                sampler=self.valid_sampler,
                                                                num_workers=4, pin_memory=True)
                self.valid_tta = True
            else:
                self.valid_set = ImageIndexDataset(REGISTRY['DATALOADER'][self.param.data_loader](self.param, split='valid'))
                self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_set)
                self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.param.batch_size,
                                                                sampler=self.valid_sampler,
                                                                num_workers=4, pin_memory=True)
                self.valid_tta = False

        with torch.no_grad():
            for i, (input, target, index) in enumerate(self.valid_loader):
                input_var = input.to(self.device)
                target_var = target.to(self.device).type(torch.long)

                if with_tta:
                    logits_merger = tta.base.Merger(type='tsharpen', n=len(tta_transforms))
                    loss_merger = tta.base.Merger(type='mean', n=len(tta_transforms))
                    for idx, t in enumerate(tta_transforms):  # custom transforms or e.g. tta.aliases.d4_transform()
                        # augment image
                        s_input_var = t.augment_image(input_var)

                        # pass to model
                        s_loss, s_logits = self.model.evaluate(s_input_var, target_var)
                        s_logits = torch.nn.functional.softmax(s_logits, dim=1)

                        # save results
                        logits_merger.append(s_logits)
                        loss_merger.append(s_loss)
                    # reduce results as you want, e.g mean/max/min
                    loss = torch.as_tensor(loss_merger.result).to(device=self.device)
                    logits = torch.as_tensor(logits_merger.result).to(device=self.device)
                else:
                    loss, logits = self.model.evaluate(input_var, target_var)
                    loss = torch.as_tensor(loss).to(device=self.device)

                prec5 = accuracy(logits, target_var, 5)
                prec1 = accuracy(logits, target_var, 1)

                loss, prec1, prec5 = \
                    self.sync_metrics(loss, prec1, prec5)
                loss, prec1, prec5 = list(map(lambda x: x.item(), (loss, prec1, prec5)))


                softmax_losses.update(loss, logits.size(0))
                top1.update(prec1, logits.size(0))
                top5.update(prec5, logits.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 20 == 0 and self.local_rank == 0:
                    print('Time: {time}\nTest: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.valid_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                        top1=top1, top5=top5, time=time.asctime(time.localtime(time.time()))))
            if self.local_rank == 0:
                print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        return top1.avg

    def save(self, epoch=0, prec1=0, is_best=False):
        save_checkpoint(
            self.name,
            self.param,
            'checkpoint-%s.pth.tar' % (epoch + 1),
            {
                'epoch': epoch + 1,
                'best_prec': prec1,
                'state_dict': self.model.state_dict()
            },
            is_best)

    def load(self, epoch=0, is_best=False):
        epoch, best_prec, model_param = \
            load_checkpoint(
                self.name,
                'checkpoint-%s.pth.tar' % (epoch + 1),
                is_best)
        self.model.load_state_dict(model_param)
        print(epoch, best_prec)



if __name__ == '__main__':
    from fgvc_cl.utils.ConfigManager import ConfigManager
    print_loaded(REGISTRY)
    c = ConfigManager()
    c.load('../../config/Default.yaml')
    c.attention_num = 2
    c.batch_size = 2
    trainer = APINetTrainer(c)
    trainer.train()
