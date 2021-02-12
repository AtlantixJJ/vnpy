import sys
sys.path.insert(0, ".")
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from lib.dataset import PointwiseDataset
from lib.models import GRUClassifier, TransformerClassifier, Learner
from lib.utils import plot_wave_pr
from lib.callback import WavePRVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="expr",
        help="The experiment directory.")
    parser.add_argument("--model", type=str, default="GRU",
        choices=["GRU", "MLP", "T"],
        help="Model type.")
    parser.add_argument("--num-layers", type=int, default=2,
        help="The number of hidden layers.")
    parser.add_argument("--hidden-size", type=int, default=64,
        help="The dimension of hidden layers.")
    parser.add_argument("--train-years", type=str, default="2013-2017",
        help="The range of training years.")
    parser.add_argument("--val-years", type=str, default="2018-2018",
        help="The range of validation years. In the same year.")
    parser.add_argument("--test-years", type=str, default="2019-2019",
        help="The range of testing years. Not in the same year.") 
    args = parser.parse_args()

    args_name = f"wavepr_{args.model}_n{args.num_layers}_h{args.hidden_size}"
    logger = pl_logger.TensorBoardLogger(args.expr, name=args_name)
    dm = PointwiseDataset(
        labels=["hold", "buy", "sell"],
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years)

    # show dataset results
    dl = dm.val_dataloader(return_info=True)
    feats, infos, labels = [], [], []
    for i, (feat, info, label) in enumerate(dl):
        feats.append(feat)
        infos.append(info)
        labels.append(label)
        plot_wave_pr(feat, info, label, label)
        plt.tight_layout()
        plt.savefig(f"results/train_viz_{i}.png")
        plt.close()
        if i > 4:
            break
    print(str(dm))
    wavepr_viz = WavePRVisualizer(feats, infos, labels)
    if args.model == "GRU":
        model = GRUClassifier(
            in_dim=5,
            n_class=3,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)
    elif args.model == "T":
        model = TransformerClassifier(
            in_dim=5,
            n_class=len(dm.labels),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)
    learner = Learner(model,
        labels=dm.labels, is_rnn=True)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[wavepr_viz],
        max_epochs=500,
        check_val_every_n_epoch=5,
        progress_bar_refresh_rate=1)
    #res = trainer.test(learner, dm.test_dataloader())
    trainer.fit(learner, dm)
    res = trainer.test(learner, dm.test_dataloader())
    torch.save(model.state_dict(), f"{args.expr}/{args_name}/model.pth")