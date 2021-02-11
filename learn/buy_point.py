import sys
sys.path.insert(0, ".")
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
import torch
import numpy as np
import argparse

from lib.dataset import BuyPoint
from lib.models import GRUClassifier, TransformerClassifier, Learner
from lib.utils import plot_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="expr",
        help="The experiment directory.")
    parser.add_argument("--model", type=str, default="GRU",
        choices=["GRU", "MLP", "T"],
        help="Model type.")
    parser.add_argument("--num-layers", type=int, default=4,
        help="The number of hidden layers.")
    parser.add_argument("--hidden-size", type=int, default=128,
        help="The dimension of hidden layers.")
    parser.add_argument("--train-years", type=str, default="2013-2017",
        help="The range of training years.")
    parser.add_argument("--val-years", type=str, default="2013-2017",
        help="The range of validation years. In the same year.")
    parser.add_argument("--test-years", type=str, default="2018-2019",
        help="The range of testing years. Not in the same year.") 
    args = parser.parse_args()

    args_name = f"buypoint_{args.model}_n{args.num_layers}_h{args.hidden_size}"
    logger = pl_logger.TensorBoardLogger(args.expr, name=args_name)
    dm = BuyPoint(
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years)

    # show dataset results
    for i, (x, y) in enumerate(dm.train_dataloader()):
        FEAT_SIZE, WIN_SIZE = x.shape[1:]
        p = x[0, 1]
        v = x[0, 4]
        marks = np.zeros((WIN_SIZE,))
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

    if args.model == "GRU":
        model = GRUClassifier(
            in_dim=FEAT_SIZE,
            n_class=len(dm.label_keys),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)
    elif args.model == "T":
        model = TransformerClassifier(
            in_dim=FEAT_SIZE,
            n_class=len(dm.label_keys),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)
    learner = Learner(model,
        labels=dm.label_keys, is_rnn=True)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=100,
        progress_bar_refresh_rate=1)
    res = trainer.test(learner, dm.val_dataloader())
    trainer.fit(learner, dm)
    res = trainer.test(learner, dm.test_dataloader())
    torch.save(model.state_dict(), f"{args.expr}/{args_name}/model.pth")