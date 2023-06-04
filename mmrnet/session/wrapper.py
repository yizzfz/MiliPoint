import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


class ModelWrapper(pl.LightningModule):

    def __init__(
            self,
            model,
            learning_rate=5e-4,
            weight_decay=1e-5,
            epochs=200,
            optimizer=None):
        super().__init__()
        self.model = model
        self.num_classes = self.model.num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if self.num_classes is None:      # keypoints
            self.loss = torch.nn.MSELoss()
            self.metric_name = 'mle'
            self.metric = mean_localization_error
        else:                           # iden or action
            self.loss = torch.nn.CrossEntropyLoss()
            self.metric_name = 'acc'
            self.metric = acc
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

        self.best_val_loss = 10e9

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)

        # loss
        self.log_dict(
            {"loss": loss, f'{self.metric_name}': metric},
            on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)
        # val_loss
        self.log_dict(
            {"val_loss": loss, f'val_{self.metric_name}': metric},
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_losses.append(loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)
        # val_loss
        if self.metric_name == 'acc':
            # compute top-3
            top3 = torch.topk(y_hat, 3, dim=1)[1]
            top3_acc = (top3 == y.unsqueeze(-1)).float().sum()/x.shape[0]
            
            self.log_dict(
                {
                    "test_loss": loss, 
                    f'test_{self.metric_name}': metric,
                    f'test_top3_{self.metric_name}': top3_acc},
                on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            self.log_dict(
                {
                    "test_loss": loss, 
                    f'test_{self.metric_name}': metric},
                on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        return self(x), y

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer in ['sgd_warmup', 'sgd']:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True)
            if self.optimizer == 'sgd':
                scheduler = CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=0.0)
        return {
            "optimizer": opt,
            "lr_scheduler":  scheduler}
    
    def on_validation_epoch_end(self):
        if len(self.val_losses):
            val_loss = torch.stack(self.val_losses).mean()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            self.val_losses = []


def mean_localization_error(x, y):
    dist = (x-y).pow(2).sum(-1).sqrt().mean()
    return dist

def acc(x, y):
    acc = (torch.argmax(x, axis=1) == y).float().sum()/x.shape[0]
    return acc
