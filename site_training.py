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
from utils.data_loader import get_trn_loader, get_val_loader, get_multi_site_val_loader, get_multi_site_trn_loader

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class MultiSiteTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, site_number=5, comment=None, layer=None, sub_layer=None, model_name=None, merge_mode=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser(description="Test training")
        parser.add_argument("--epochs", default=6, type=int, help="number of training epochs")
        parser.add_argument("--batch_size", default=512, type=int, help="number of batch size")
        parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--in_channels", default=3, type=int, help="number of image channels")
        parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
        parser.add_argument("--site_number", default=5, type=int, help="number of sites taking part in learning")
        parser.add_argument("--layer", default=None, type=str, help="layer in which training should take place")
        parser.add_argument("--model_name", default='resnet', type=str, help="name of model to use")
        parser.add_argument("--merge_mode", default='projection', type=str, help="describes which parameters of the model to merge")
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
        if site_number is not None:
            self.args.site_number = site_number
        if comment is not None:
            self.args.comment = comment
        if layer is not None:
            self.args.layer = layer
        if sub_layer is not None:
            self.args.sub_layer = sub_layer
        if model_name is not None:
            self.args.model_name = model_name
        if merge_mode is not None:
            self.args.merge_mode = merge_mode
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.models= self.initModel()
        self.optims = self.initOptimizer()

    def initModel(self):
        models = []
        for i in range(self.args.site_number):
            models.append(AttUNet(num_classes=90))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                for model in models:
                    model = nn.DataParallel(model)
            for model in models:
                model = model.to(self.device)
        return models

    def initOptimizer(self):
        optims = []
        for i in range(self.args.site_number):
            optims.append(Adam(params=self.models[i].parameters(), lr=self.args.lr))

        return optims

    def initDl(self):
        trn_dls = []
        for i in range(self.args.site_number):
            trn_dls.append(get_trn_loader(self.args.batch_size, site=i, site_number=self.args.site_number))
        val_dl = get_val_loader(self.args.batch_size)

        multi_trn_dl = get_multi_site_trn_loader(self.args.batch_size, site_number=self.args.site_number)
        multi_val_dl = get_multi_site_val_loader(self.args.batch_size, site_number=self.args.site_number)

        return trn_dls, val_dl, multi_trn_dl, multi_val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        trn_dls, val_dl, multi_trn_dl, multi_val_dl = self.initDl()

        saving_criterion = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):
            
            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.args.epochs,
                    len(trn_dls[0]),
                    len(val_dl),
                    self.args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trnMetrics = torch.zeros(5, self.args.site_number,  len(trn_dls[0].dataset), device=self.device)

            trnMetrics = self.doMultiTraining(epoch_ndx, multi_trn_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, correct_ratio = self.doMultiValidation(epoch_ndx, multi_val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics)
                saving_criterion = max(correct_ratio, saving_criterion)

                self.saveModel('mnist', epoch_ndx, correct_ratio == saving_criterion)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doMultiTraining(self, epoch_ndx, multi_trn_dl):
        trnMetrics = torch.zeros(3, self.args.site_number, len(multi_trn_dl.dataset), device=self.device)

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            log.warning('E{} Training ---/{} starting'.format(epoch_ndx, len(multi_trn_dl)))

        for batch_ndx, batch_tuples in enumerate(multi_trn_dl):
            for model in self.models:
                model.train()

            for optim in self.optims:
                optim.zero_grad()

            loss, _ = self.computeMultiBatchLoss(
                batch_ndx,
                batch_tuples,
                trnMetrics,
                'trn')
            loss.sum().backward()
            
            for optim in self.optims:
                optim.step()

            if self.args.merge_mode == 'projection':
                self.mergeParams(layer_names=['qkv'], depth=1)
            elif self.args.merge_mode == 'second_half':
                self.mergeParams(layer_names=['block3', 'block4', 'lin'], depth=0)
            elif self.args.merge_mode == 'first_half':
                self.mergeParams(layer_names=['block1', 'block2', 'conv0'], depth=0)

        self.totalTrainingSamples_count += len(multi_trn_dl.dataset) * self.args.site_number

        return trnMetrics.to('cpu')

    def doMultiValidation(self, epoch_ndx, multi_val_dl):
        with torch.no_grad():
            valMetrics = torch.zeros(3, self.args.site_number, len(multi_val_dl.dataset), device=self.device)
            for model in self.models:
                model.eval()

            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(multi_val_dl)))

            for batch_ndx, batch_tuples in enumerate(multi_val_dl):

                loss, accuracy = self.computeMultiBatchLoss(
                    batch_ndx,
                    batch_tuples,
                    valMetrics,
                    'val'
                )

        return valMetrics.to('cpu'), accuracy

    def computeMultiBatchLoss(self, batch_ndx, batch_tups, metrics, mode):
        batches, mask_batches, img_ids = batch_tups
        batches = batches.to(device=self.device, non_blocking=True).permute(1, 0, 2, 3, 4)
        mask_batches = mask_batches.to(device=self.device, non_blocking=True).permute(1, 0, 2, 3)


        if mode == 'trn':
            angle = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            scale = random.uniform(0.9, 1.1)
            for batch in batches:
                batch = functional.rotate(batch, angle)
            if flip:
                for batch in batches:
                    batch = functional.hflip(batch)
            batches = scale * batches

        preds = torch.zeros((batches.shape[0], batches.shape[1], 90, 64, 64)).to(device=self.device)
        for i in range(self.args.site_number):
            batch = batches[i]
            preds[i] = self.models[i](batch)
        pred_labels = torch.argmax(preds, dim=2)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss_pxwise = loss_fn(preds.flatten(start_dim=0, end_dim=1), mask_batches.flatten(start_dim=0, end_dim=1))
        loss = loss_pxwise.sum(dim=[1, 2])

        correct_mask = pred_labels == mask_batches
        background_mask = mask_batches == 0
        correct_mask[background_mask] = False

        pos_pred = pred_labels > 0
        neg_pred = ~pos_pred
        pos_mask = mask_batches > 0
        neg_mask = ~pos_mask

        neg_count = neg_mask.sum(dim=[2, 3]).int()
        pos_count = pos_mask.sum(dim=[2, 3]).int()

        true_pos = (pos_pred & pos_mask).sum(dim=[2, 3]).int()
        true_neg = (neg_pred & neg_mask).sum(dim=[2, 3]).int()
        false_pos = neg_count - true_neg
        false_neg = pos_count - true_pos
        epsilon = 0.1

        dice_score = (2*true_pos + epsilon) / (2*true_pos + false_pos + false_neg + epsilon)

        correct = torch.sum(correct_mask, dim=[2, 3])
        pos_pred_count = torch.sum(pos_pred, dim=[2, 3])
        accuracy = (correct+epsilon) / (pos_pred_count+epsilon)

        start_ndx = batch_ndx * self.args.batch_size
        end_ndx = start_ndx + mask_batches.size(1)

        metrics[0, :, start_ndx:end_ndx] = loss.detach().view(batches.shape[0:2])
        metrics[1, :, start_ndx:end_ndx] = accuracy
        metrics[2, :, start_ndx:end_ndx] = dice_score
        return loss, accuracy.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics
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
        for i in range(self.args.site_number):
            writer.add_scalar(
                'loss/site {}'.format(i + 1),
                scalar_value=metrics[0, i, :].mean(),
                global_step=self.totalTrainingSamples_count
            )
            writer.add_scalar(
                'accuracy/site {}'.format(i + 1),
                scalar_value=metrics[1, i, :].mean(),
                global_step=self.totalTrainingSamples_count
            )
            writer.add_scalar(
                'dice score/site {}'.format(i + 1),
                scalar_value=metrics[2, i, :].mean(),
                global_step=self.totalTrainingSamples_count
            )
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

        model = self.models[0]
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optims[0].state_dict(),
            'optimizer_name': type(self.optims[0]).__name__,
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

    def mergeParams(self, layer_names=None, depth=None):
        dicts = []
        for i in range(self.args.site_number):
            dicts.append(self.models[i].state_dict())
        dict_avg = {}

        names = self.models[0].named_parameters()

        for name, _ in names:
            if self.args.model_name == 'resnet':
                layer = name.split('.')[1]
                sub_layer = name.split('.')[2]
                if (layer != self.args.layer or sub_layer != self.args.sub_layer) and (layer != 'conv1') and (layer != 'fc'):
                    dict_avg[name] = torch.zeros(dicts[0][name].shape, device=self.device)
                    for i in range(self.args.site_number):
                        dict_avg[name] += dicts[i][name]
                    dict_avg[name] = dict_avg[name] / self.args.site_number
            if self.args.model_name == 'unet':
                layer = name.split('.')[depth]
                if layer in layer_names:
                    dict_avg[name] = torch.zeros(dicts[0][name].shape, device=self.device)
                    for i in range(self.args.site_number):
                        dict_avg[name] += dicts[i][name]
                    dict_avg[name] = dict_avg[name] / self.args.site_number

        for i in range(self.args.site_number):
            self.models[i].load_state_dict(dict_avg, strict=False)


if __name__ == '__main__':
    MultiSiteTrainingApp().main()
