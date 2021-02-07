import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import numpy as np
import os
from pytorch_lightning.metrics.functional import precision_recall


class MLP(pl.LightningModule):
    def __init__(self, in_dim=5, n_class=4, labels=[], dims=[]):
        super().__init__()
        self.n_layers = len(dims) + 1
        self.dims = dims
        self.in_dim = in_dim
        self.n_class = n_class
        self.labels = labels
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.build()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def build(self):
        self.layers = torch.nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(torch.nn.Linear(self.in_dim, self.n_class))
        else:
            self.layers.append(torch.nn.Linear(self.in_dim, self.dims[0]))
            self.layers.append(torch.nn.ReLU(inplace=True))

        for i in range(self.n_layers - 2):
            layer = torch.nn.Linear(self.dims[i], self.dims[i + 1])
            self.layers.append(layer)
            act = torch.nn.ReLU(inplace=True)
            self.layers.append(act)
        
        if self.n_layers > 1:
            self.layers.append(torch.nn.Linear(
                self.dims[-1], self.n_class))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y_true = batch # (32, 10, 5)
        x = x.view(x.shape[0], -1).float() - 1
        y = self(x)
        c = self.loss_fn(y, y_true.long())
        P, R = precision_recall(y.argmax(1), y_true,
                num_classes=len(self.labels),
                class_reduction="none")
        for i in range(len(self.labels)):
            self.log(f'train/P/{self.labels[i]}', P[i],
                on_step=False, on_epoch=True)
            self.log(f'train/R/{self.labels[i]}', R[i],
                on_step=False, on_epoch=True)
        self.log(f'train/P', P.mean(),
            on_step=False, on_epoch=True)
        self.log(f'train/R', R.mean(),
            on_step=False, on_epoch=True)
        return c

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y = self(x.view(x.shape[0], -1).float() - 1).argmax(1)
        P, R = precision_recall(y, y_true,
            num_classes=len(self.labels),
            class_reduction="none")
        for i in range(len(self.labels)):
            self.log(f'val/P/{self.labels[i]}', P[i],
                on_step=False, on_epoch=True)
            self.log(f'val/R/{self.labels[i]}', R[i],
                on_step=False, on_epoch=True)
        self.log(f'val/P', P.mean(),
            on_step=False, on_epoch=True)
        self.log(f'val/R', R.mean(),
            on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y = self(x.view(x.shape[0], -1).float() - 1).argmax(1)
        P, R = precision_recall(y, y_true,
            num_classes=len(self.labels),
            class_reduction="none")
        for i in range(len(self.labels)):
            self.log(f'test/P/{self.labels[i]}', P[i],
                on_step=False, on_epoch=True)
            self.log(f'test/R/{self.labels[i]}', R[i],
                on_step=False, on_epoch=True)
        self.log(f'test/P', P.mean(),
            on_step=False, on_epoch=True)
        self.log(f'test/R', R.mean(),
            on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())