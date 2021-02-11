import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import numpy as np
import os
from pytorch_lightning.metrics.functional import precision_recall


class TransformerClassifier(torch.nn.Module):
    def __init__(self, in_dim=5, n_class=4,
                 hidden_size=128, num_layers=2, nhead=8):
        super().__init__()
        self.in_dim = in_dim
        self.n_class = n_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed_layer = MLP(
            in_dim=in_dim,
            n_class=hidden_size,
            dims=[32] * 4)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=32,
            dim_feedforward=hidden_size,
            dropout=0.1,
            nhead=nhead)
        self.enc = torch.nn.TransformerEncoder(enc_layer,
            num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_size, n_class)
    
    def forward(self, x):
        L, N, C = x.shape
        x = self.embed_layer(x.view(-1, C)).view(L, N, -1)
        outputs = self.enc(x) # (T, N, E)
        print(outputs.shape)
        return self.linear(outputs.mean(0))


class GRUClassifier(torch.nn.Module):
    def __init__(self, in_dim=5, n_class=4,
                 hidden_size=128, num_layers=2):
        super().__init__()
        self.in_dim = in_dim
        self.n_class = n_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = torch.nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers)
        self.act = torch.nn.ReLU()
        self.linear = torch.nn.Linear(hidden_size, n_class)
    
    def forward(self, x):
        outputs, hiddens = self.gru(x)
        L, N, C = outputs.shape
        return self.linear(outputs.view(-1, C)).view(L, N, -1)


class MLP(torch.nn.Module):
    def __init__(self, in_dim=5, n_class=4, dims=[]):
        super().__init__()
        self.n_layers = len(dims) + 1
        self.dims = dims
        self.in_dim = in_dim
        self.n_class = n_class
        self.build()

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


class MLP(torch.nn.Module):
    def __init__(self, in_dim=5, n_class=4, dims=[]):
        super().__init__()
        self.n_layers = len(dims) + 1
        self.dims = dims
        self.in_dim = in_dim
        self.n_class = n_class
        self.build()

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


class Learner(pl.LightningModule):
    def __init__(self, model, labels=[], is_rnn=False):
        super().__init__()
        self.model = model
        self.n_class = len(labels)
        self.labels = labels
        self.is_rnn = is_rnn
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
    
    def training_step(self, batch, batch_idx):
        x, y_true = batch # N, L, C
        y_true = y_true.permute(1, 0).reshape(-1)
        if self.is_rnn:
            x = x.permute(1, 0, 2)
        else:
            x = x.view(x.shape[0], -1)
        y = self.model(x)
        c = self.loss_fn(y.view(-1, y.shape[-1]), y_true)
        P, R = precision_recall(y.argmax(2).view(-1), y_true,
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
        y_true = y_true.permute(1, 0).reshape(-1)
        if self.is_rnn:
            x = x.permute(1, 0, 2)
        else:
            x = x.view(x.shape[0], -1)
        y = self.model(x).argmax(2)
        P, R = precision_recall(y.view(-1), y_true,
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
        x, y_true = batch # (32, 5, 10)
        y_true = y_true.permute(1, 0).reshape(-1)
        if self.is_rnn:
            x = x.permute(1, 0, 2)
        else:
            x = x.view(x.shape[0], -1)
        y = self.model(x).argmax(2)
        P, R = precision_recall(y.view(-1), y_true,
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