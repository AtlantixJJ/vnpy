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
from lib.utils import plot_multicolor_line, label2color

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

    args_name = f"wavepr_{args.model}_n{args.num_layers}_h{args.hidden_size}"
    logger = pl_logger.TensorBoardLogger(args.expr, name=args_name)
    dm = PointwiseDataset(
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years)

    # show dataset results
    for i, (feats, infos, labels) in enumerate(dm.train_dataloader()):
        colors = label2color(labels[0])
        x = np.arange(feats.shape[1])
        close_prices = feats[0, :, 1].numpy()
        fig = plt.figure(figsize=(10, 7))
        ax1 = plt.subplot(2, 1, 1)
        plot_multicolor_line(ax1, x, close_prices, colors)
        ax1.set_xlim([0, x.shape[0]])
        ax1.set_ylim([-10, 10])
        ax1.set_title("Delta Percentages (%) & Wave")
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(infos[0, :, 0] * 100)
        ax2.plot(infos[0, :, 1] * 100)
        ax2.set_title("Wave Profit & Retract (%)")
        plt.savefig(f"results/train_viz_{i}.png")
        plt.close()
        if i > 4:
            break
    print(str(dm))

    if args.model == "GRU":
        model = GRUClassifier(
            in_dim=5,
            n_class=3,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)
    elif args.model == "T":
        model = TransformerClassifier(
            in_dim=5,
            n_class=3,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers)
    learner = Learner(model,
        labels=["hold", "buy", "sell"], is_rnn=True)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=100,
        progress_bar_refresh_rate=1)
    res = trainer.test(learner, dm.val_dataloader())
    trainer.fit(learner, dm)
    res = trainer.test(learner, dm.test_dataloader())
    torch.save(model.state_dict(), f"{args.expr}/{args_name}/model.pth")