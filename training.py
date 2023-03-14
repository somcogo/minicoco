import sys
import argparse
import os
import datetime
import shutil
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.transforms import functional

from models.model import AttUNet
from utils.logconf import logging
from utils.data_loader import get_trn_loader, get_val_loader

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class MiniCocoTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, site=None, comment=None, site_number=5, model_name=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser(description="Test training")
        parser.add_argument("--epochs", default=2, type=int, help="number of training epochs")
        parser.add_argument("--batch_size", default=500, type=int, help="number of batch size")
        parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--in_channels", default=3, type=int, help="number of image channels")
        parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
        parser.add_argument("--site", default=None, type=int, help="index of site to train on")
        parser.add_argument("--model_name", default='resnet', type=str, help="name of the model to use")
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')

        self.args = parser.parse_args()
        if epochs is not None:
            self.args.epochs = epochs
        if batch_size is not None:
            self.args.batch_size = batch_size
        if logdir is not None:
            self.args.logdir = logdir
        if lr is not None:
            self.args.lr = lr
        if site is not None:
            self.args.site = site
        if comment is not None:
            self.args.comment = comment
        if site_number is not None:
            self.args.site_number = site_number
        if model_name is not None:
            self.args.model_name = model_name
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = AttUNet(num_classes=90)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return Adam(params=self.model.parameters(), lr=self.args.lr)

    def initDl(self):
        trn_dl = get_trn_loader(self.args.batch_size, site=self.args.site)
        val_dl = get_val_loader(self.args.batch_size)
        return trn_dl, val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        train_dl, val_dl = self.initDl()

        saving_criterion = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):

            
            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.args.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trnMetrics = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, correct_ratio = self.doValidation(epoch_ndx, val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics)
                saving_criterion = max(correct_ratio, saving_criterion)

                self.saveModel('imagenet', epoch_ndx, correct_ratio == saving_criterion)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics = torch.zeros(3, len(train_dl.dataset), device=self.device)

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            log.warning('E{} Training ---/{} starting'.format(epoch_ndx, len(train_dl)))

        for batch_ndx, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss, _ = self.computeBatchLoss(
                batch_ndx,
                batch_tuple,
                trnMetrics,
                'trn')

            loss.backward()
            self.optimizer.step()

            if batch_ndx % 100 == 0 and batch_ndx > 99:
                log.info('E{} Training {}/{}'.format(epoch_ndx, batch_ndx, len(train_dl)))

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics = torch.zeros(3, len(val_dl.dataset), device=self.device)

            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(val_dl)))

            for batch_ndx, batch_tuple in enumerate(val_dl):
                _, correct_ratio = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    valMetrics,
                    'val'
                )
                if batch_ndx % 50 == 0 and batch_ndx > 49:
                    log.info('E{} Validation {}/{}'.format(epoch_ndx, batch_ndx, len(val_dl)))

        return valMetrics.to('cpu'), correct_ratio

    def computeBatchLoss(self, batch_ndx, batch_tup, metrics, mode):
        batch, mask_batch, img_ids = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        mask_batch = mask_batch.to(device=self.device, non_blocking=True)

        if mode == 'trn':
            angle = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            scale = random.uniform(0.9, 1.1)
            batch = functional.rotate(batch, angle)
            if flip:
                batch = functional.hflip(batch)
            batch = scale * batch

        pred = self.model(batch)
        pred_label = torch.argmax(pred, dim=1)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss_ptwise = loss_fn(pred, mask_batch)
        loss = loss_ptwise.sum(dim=[1, 2])

        correct_mask = pred_label == mask_batch
        background_mask = mask_batch == 0
        correct_mask[background_mask] = False

        pos_pred = pred_label > 0
        neg_pred = ~pos_pred
        pos_mask = mask_batch > 0
        neg_mask = ~pos_mask

        neg_count = neg_mask.sum(dim=[1, 2]).int()
        pos_count = pos_mask.sum(dim=[1, 2]).int()

        true_pos = (pos_pred & pos_mask).sum(dim=[1, 2]).int()
        true_neg = (neg_pred & neg_mask).sum(dim=[1, 2]).int()
        false_pos = neg_count - true_neg
        false_neg = pos_count - true_pos
        epsilon = 0.1

        dice_score = (2*true_pos + epsilon) / (2*true_pos + false_pos + false_neg + epsilon)

        correct = torch.sum(correct_mask, dim=[1, 2])
        pos_pred_count = torch.sum(pos_pred, dim=[1, 2])
        accuracy = (correct+epsilon) / (pos_pred_count+epsilon)

        start_ndx = batch_ndx * self.args.batch_size
        end_ndx = start_ndx + mask_batch.size(0)

        metrics[0, start_ndx:end_ndx] = loss.detach()
        metrics[1, start_ndx:end_ndx] = accuracy
        metrics[2, start_ndx:end_ndx] = dice_score

        return loss.mean(), accuracy.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        img_list=None
    ):
        self.initTensorboardWriters()

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            log.info(
                "E{} {}:{} loss".format(
                    epoch_ndx,
                    mode_str,
                    metrics[0].mean()
                )
            )

        writer = getattr(self, mode_str + '_writer')
        writer.add_scalar(
            'loss/overall',
            scalar_value=metrics[0].mean(),
            global_step=self.totalTrainingSamples_count
        )
        writer.add_scalar(
            'accuracy/overall',
            scalar_value=metrics[1].mean(),
            global_step=self.totalTrainingSamples_count
        )
        writer.add_scalar(
            'dice score/overall',
            scalar_value=metrics[2].mean(),
            global_step=self.totalTrainingSamples_count
            )
        writer.flush()

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'saved_models',
            self.args.logdir,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.args.comment,
                self.totalTrainingSamples_count
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count
        }

        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'saved_models',
                self.args.logdir,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.args.comment,
                    'best'
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

if __name__ == '__main__':
    MiniCocoTrainingApp().main()
