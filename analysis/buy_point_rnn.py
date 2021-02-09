import sys
sys.path.insert(0, ".")
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
import torch
import numpy as np

from lib.dataset import BuyPoint
from lib.models import GRUClassifier, Learner
from lib.utils import plot_dict

DIR = "expr"
WIN_SIZE = 10
FEAT_SIZE = 5
dm = BuyPoint()
logger = pl_logger.TensorBoardLogger(DIR)
marks = np.ones((WIN_SIZE,))
for i, (x, y) in enumerate(dm.train_dataloader()):
    #print(x[0], y[:5], x.shape)
    p = x[0, 1]
    v = x[0, 4]
    marks.fill(0)
    a = v[1:] < v[:-1]
    marks[:a.shape[0]][a] = 1
    plot_dict({
        f"price + {int(y[0])}" : {"value": p, "chart": "line"},
        "volume" : {"value": v, "chart": "bar", "mark": marks}
    }, f"res{i}.png", x.shape[2])
    if i > 4:
        break
print(str(dm))
# batch_first=False
model = GRUClassifier(in_dim=FEAT_SIZE, hidden_size=128, num_layers=2)
learner = Learner(model, labels=dm.label_keys, is_rnn=True)
trainer = pl.Trainer(
    max_epochs=100,
    progress_bar_refresh_rate=1,
    track_grad_norm=2)
res = trainer.test(learner, dm.val_dataloader())
trainer.fit(learner, dm)
res = trainer.test(learner, dm.test_dataloader())