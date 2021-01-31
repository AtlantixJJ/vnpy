import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import numpy as np


def split(data, train_ratio, val_ratio, seed=None):
    """Split a data into train / val / test split."""
    N = data.shape[0]
    test_ratio = 1 - train_ratio - val_ratio
    rng = np.random.RandomState(seed if seed else 1)
    indice = np.arange(N)
    rng.shuffle(indice)
    st, ed = 0, int(N * train_ratio)
    train_indice = indice[:ed]
    st, ed = ed, int(N * (train_ratio + val_ratio))
    val_indice = indice[st:ed]
    test_indice = indice[ed:]
    return data[train_indice], data[val_indice], data[test_indice]


def tensor_from_dict2d(dic, keys1, keys2):
    """Create a sequential Tensor from dic, given keys."""
    data = []
    for k1 in keys1:
        for k2 in keys2:
            if k2 in dic[k1]:
                data.append(dic[k1][k2])
    return torch.from_numpy(np.concatenate(data))



class BuyPoint(pl.LightningDataModule):
    def __init__(self, fpath="data/buy_point.npy"):
        """Create training and validation split from raw data.
        Args:
            dic : dic[symbol][year]
        """
        self.dic = np.load(fpath, allow_pickle=True)[()]
        self.buy_dic =  self.dic["buy"]
        self.sell_dic = self.dic["sell"]
        self.hold_dic = self.dic["hold"]

        self.all_symbols = np.array(list(self.buy_dic.keys()))
        #self.all_years = np.array(list(self.buy_dic["600586.SSE"].keys()))
        N = self.all_symbols.shape[0]
        #M = self.all_years.shape[0]

        self.train_symbols, self.val_symbols, self.test_symbols = \
            split(self.all_symbols, 0.6, 0.2)
        
        self.train_times = ["2000"]
        self.val_times = ["2001"]
        self.test_times = ["2002"]
        self.create_datasets()


    def create_datasets(self):
        for split in ["train", "val", "test"]:
            x = []
            for i, k in enumerate(self.dic.keys()):
                val = tensor_from_dict2d(
                    dic=self.dic[k],
                    keys1=getattr(self, f"{split}_symbols"),
                    keys2=getattr(self, f"{split}_years"))
                x.append(val)
                y.append(torch.Tensor(val.shape[:1]).fill_(i))
            x = torch.cat(x)
            y = torch.cat(y)
            print(x.shape, y.shape)

        
    
    def __str__(self):
        print("=> Buy Point Dataset")
        print(f"=> Symbols number: {len(self.all_symbols)}, train ({len(self.train_symbols)}) : val ({len(self.val_symbols)} : test ({len(self.test_symbols)})")
