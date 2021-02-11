import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import numpy as np
import glob

from lib.utils import *

# test dataset: categorize by symbol and year


def label_wavepr(arr, T1=0.30, T2=0.15):
    """Label the wave profit retract to discrete labels.
    Args:
        arr: (N, 2) array, denoting profit and retract.
        T1: The threshold for minimum profit.
        T2: THe threshold for maximum retract.
    """
    P, R = arr[:, 0], arr[:, 1]
    buy_labels = (P > T1) & (-R < P / 2) # label 1
    # sell point: retracts is nonnegligiable and is larger than profits
    sell_labels = (-R > T2) & (-R > P / 2)  # label 2
    # hold point: other cases. label 0
    label = np.zeros((arr.shape[0],), dtype="uint8")
    label[buy_labels] = 1
    label[sell_labels] = 2
    return label


class DictListSampler(torch.utils.data.Dataset):
    """The data is organized in a dict.
    The value is an array that needs to be sampled."""

    def __init__(self, data_dic, label_dic, listkeys):
        self.data_dic = data_dic
        self.label_dic = label_dic
        self.listkeys = listkeys
        self.return_info = False
    
    def set_window(self, size):
        self.win_size = size
    
    def query(self, keys):
        res = [0, 0]
        if len(keys) == 1:
            res[0] = self.data_dic[keys[0]]
            res[1] = self.label_dic[keys[0]]
        elif len(keys) == 2:
            res[0] = self.data_dic[keys[0]][keys[1]]
            res[1] = self.label_dic[keys[0]][keys[1]]
        elif len(keys) == 3:
            res[0] = self.data_dic[keys[0]][keys[1]][keys[2]]
            res[1] = self.label_dic[keys[0]][keys[1]][keys[2]]
        return res
    
    def __len__(self):
        return len(self.listkeys)

    def __getitem__(self, idx):
        keys = self.listkeys[idx]
        x, info = self.query(keys)
        N = x.shape[0] # total length
        # randomly sample a position
        idx = np.random.randint(0, N - self.win_size)
        segx = x[idx : idx + self.win_size]
        segx[1:] = (segx[1:] / (1e-9 + segx[:-1]) - 1) * 100 # inc & dec
        segx[0] = 0
        seginfo = info[idx : idx + self.win_size]
        if self.return_info:
            return segx, seginfo, label_wavepr(seginfo)
        return segx, label_wavepr(seginfo)


def year_symbol_list(years, dic, min_size=50):
    res = []
    for y in years:
        for s in dic[y]:
            if dic[y][s].shape[0] > min_size:
                res.append([y, s])
    return res


class PointwiseDataset(pl.LightningDataModule):
    def __init__(self,
                 year_data_path="data/year_data.npz",
                 wave_pr_path="data/wave_pr.npz",
                 train_years="2000-2005",
                 val_years="2000-2005",
                 test_years="2006-2006"):
        super().__init__()
        self.train_years = parse_years(train_years)
        self.val_years = parse_years(val_years)
        self.test_years = parse_years(test_years)
        self.all_years = list(set(self.train_years + \
            self.val_years + self.test_years))
        
        self.data_dic = np.load(year_data_path,
                                allow_pickle=True)['arr_0'][()]
        self.label_dic = np.load(wave_pr_path,
                                allow_pickle=True)['arr_0'][()]
        win_size = 128 
        for split in ["train", "val", "test"]:
            year = getattr(self, f"{split}_years")
            l = year_symbol_list(year, self.data_dic, win_size)
            ds = DictListSampler(self.data_dic, self.label_dic, l)
            ds.set_window(win_size)
            setattr(self, f"{split}_ds", ds)

    def train_dataloader(self, batch_size=64, return_info=False):
        self.train_ds.return_info = return_info
        return DataLoader(self.train_ds,
            batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self, batch_size=64):
        self.val_ds.return_info = return_info
        return DataLoader(self.val_ds,
            batch_size=batch_size, shuffle=False)
    
    def test_dataloader(self, batch_size=64):
        self.test_ds.return_info = return_info
        return DataLoader(self.test_ds,
            batch_size=batch_size, shuffle=False)
            

class BuyPoint(pl.LightningDataModule):
    def __init__(self,
                 data_dir="data/buy_point",
                 train_years="2000-2005",
                 val_years="2000-2005",
                 test_years="2006-2006"):
        """Create training and validation split from raw data.
        Args:
            dic : dic[symbol][year]
        """
        super().__init__()
        self.dic = {}
        self.train_years = parse_years(train_years)
        self.val_years = parse_years(val_years)
        self.test_years = parse_years(test_years)
        self.all_years = list(set(self.train_years + \
            self.val_years + self.test_years))

        load_year(self.dic, data_dir, years=self.all_years)
        self.label_keys = list(self.dic.keys())
        k = self.label_keys[0]
        self.all_symbols = np.array(list(self.dic[k].keys()))
        N_symbols = self.all_symbols.shape[0]
        self.N_symbols = N_symbols

        self.train_symbols, self.val_symbols, self.test_symbols = \
            split(self.all_symbols, 0.7, 0.3)
        self.test_symbols = self.all_symbols
        # This test is not formal

        self.create_datasets()

    def create_datasets(self):
        for split in ["train", "val", "test"]:
            x, y = [], []
            for i, k in enumerate(self.label_keys):
                val = tensor_from_dict2d(
                    dic=self.dic[k],
                    keys1=getattr(self, f"{split}_symbols"),
                    keys2=getattr(self, f"{split}_years"),
                    delete=True)
                if val is not None:
                    x.append(val)
                    y.append(torch.Tensor(val.shape[:1]).fill_(i))

            if len(x) > 0:
                x, y = balance_class(x, y)
                for i in range(len(y)):
                    setattr(self,
                        f"N_{split}_{self.label_keys[i]}",
                        y[i].shape[0])
                x = torch.cat(x)
                y = torch.cat(y)
                setattr(self, f"N_{split}", x.shape[0])
                setattr(self, f"{split}_ds", TensorDataset(x, y))
            else:
                print("!> Empty dataset!")
                setattr(self, f"N_{split}", 0)

        self.N = self.N_train + self.N_val + self.N_test
        self.train_data_ratio = float(self.N_train) / self.N * 100
        self.val_data_ratio = float(self.N_val) / self.N * 100
        self.test_data_ratio = float(self.N_test) / self.N * 100

    def __str__(self):
        s = "=> Buy Point Dataset\n"
        s += f"=> num classes is {len(self.label_keys)}"
        t = ",".join(self.label_keys)
        s += f", {t}\n"
        r1 = self.train_data_ratio
        r2 = self.val_data_ratio
        r3 = self.test_data_ratio
        s += f"=> Total: {self.N}, train({r1:.2f}%) : val({r2:.2f}%) : test({r3:.2f}%)\n"

        t1 = ""
        for split in ["train", "val", "test"]:
            N = getattr(self, f"N_{split}")
            t = ""
            for i, k in enumerate(self.label_keys):
                if not hasattr(self, f"N_{split}_{k}"):
                    continue
                M = getattr(self, f"N_{split}_{k}")
                r = float(M) / N * 100
                t += f"{k}({M}) {r:.2f}%, "
            t1 += f"=> {split}({N}): " +  t[:-2] + "\n"
        s += t1
        return s

    def train_dataloader(self, batch_size=64):
        return DataLoader(self.train_ds,
            batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self, batch_size=64):
        return DataLoader(self.val_ds,
            batch_size=batch_size, shuffle=False)
    
    def test_dataloader(self, batch_size=64):
        return DataLoader(self.test_ds,
            batch_size=batch_size, shuffle=False)