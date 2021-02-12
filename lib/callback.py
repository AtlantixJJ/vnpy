import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from lib.utils import plot_wave_pr

class WavePRVisualizer(pl.Callback):
  def __init__(self, feats, infos, labels):
    self.feats = feats
    self.infos = infos
    self.labels = labels
    self.count = 0

  def on_validation_end(self, trainer, pl_module):
    # assume each is of batch size 1
    idx = 0
    for feat, info, y_true in zip(self.feats, self.infos, self.labels): 
      N, L, C = feat.shape
      y = pl_module(feat).permute(1, 0, 2)
      y_pred = y.argmax(2)
      fig = plot_wave_pr(feat, info, y_true, y_pred)
      trainer.logger.experiment.add_figure(f"Wave PR Prediction {idx}",
        fig, self.count)
      plt.close()
      idx += 1
    self.count += 1
