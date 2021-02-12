import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import numpy as np
import glob

from lib.utils import *


def year_symbol_list(years, dic, min_size=50):
    res = []
    for y in years:
        for s in dic[y]:
            if dic[y][s].shape[0] > min_size:
                res.append([y, s])
    return res


def label_wavepr(arr, T1=0.30, T2=0.15):
    """Label the wave profit retract to discrete labels.
    Args:
        arr: (N, 2) array, denoting profit and retract.
        T1: The threshold for minimum profit.
        T2: The threshold for maximum retract.
    """
    P, R = arr[:, 0], arr[:, 1]
    buy_labels = (P > T1) & (-R < P / 2) # label 1
    # sell point: retracts is nonnegligiable and is larger than profits
    sell_labels = (-R > T2) & (-R > P / 2)  # label 2
    # hold point: other cases. label 0
    label = np.zeros((arr.shape[0],), dtype="uint8")
    label[buy_labels] = 1
    label[sell_labels] = 2
    buy_count = buy_labels.sum()
    sell_count = sell_labels.sum()
    N = arr.shape[0]
    return label, [N - buy_count - sell_count, buy_count, sell_count]


def label_wavepr_dataset(year_symbols, label_dic, info_dic):
    """Label a whole year-symbol dataset.
    Args:
        year_symbols: The key list.
        label_dic: The target dict to be modified.
        info_dic: The information dict used for labeling.
    Returns:
        The label count for each category.
    """
    total_counts = []
    for k1, k2 in year_symbols:
        if k1 not in label_dic:
            label_dic[k1] = {}
        label, counts = label_wavepr(info_dic[k1][k2])
        label_dic[k1][k2] = label
        if len(total_counts) == 0:
            total_counts = np.array(counts)
        else:
            total_counts += np.array(counts)
    return total_counts


class DictListSampler(torch.utils.data.Dataset):
    """The data is organized in a dict.
    The value is an array that needs to be sampled.
    """

    def __init__(self, listkeys, data_dic, label_dic, info_dic=None):
        self.data_dic = data_dic
        self.label_dic = label_dic
        self.info_dic = info_dic
        self.listkeys = listkeys
        self.return_info = False
    
    def set_window(self, size):
        self.win_size = size
    
    def query(self, keys):
        res = [0, 0, 0]
        if len(keys) == 1:
            res[0] = self.data_dic[keys[0]]
            res[1] = self.info_dic[keys[0]]
            res[2] = self.label_dic[keys[0]]
        elif len(keys) == 2:
            res[0] = self.data_dic[keys[0]][keys[1]]
            res[1] = self.info_dic[keys[0]][keys[1]]
            res[2] = self.label_dic[keys[0]][keys[1]]
        elif len(keys) == 3:
            res[0] = self.data_dic[keys[0]][keys[1]][keys[2]]
            res[1] = self.info_dic[keys[0]][keys[1]][keys[2]]
            res[2] = self.label_dic[keys[0]][keys[1]][keys[2]]
        return res
    
    def __len__(self):
        return len(self.listkeys)

    def __getitem__(self, idx):
        keys = self.listkeys[idx]
        x, info, label = self.query(keys)
        N = x.shape[0] # total length
        # randomly sample a position
        if self.win_size == 0:
            segx = torch.FloatTensor(x)
            seginfo = info
            seglabel = label
        else:
            idx = np.random.randint(0, N - self.win_size)
            segx = torch.FloatTensor(x[idx : idx + self.win_size])
            seginfo = info[idx : idx + self.win_size]
            seglabel = label[idx : idx + self.win_size]

        segx[1:] = (segx[1:] / (1e-9 + segx[:-1]) - 1) * 100 
        segx[0] = 0
        seglabel = torch.LongTensor(seglabel)

        if self.return_info and self.info_dic:
            return segx, torch.FloatTensor(seginfo), seglabel
        return segx, seglabel


class PointwiseDataset(pl.LightningDataModule):
    """year data and corresponding pointwise labeling."""

    def __init__(self,
                 labels=["hold", "buy", "sell"],
                 year_data_path="data/year_data.npz",
                 wave_pr_path="data/wave_pr.npz",
                 train_years="2000-2005",
                 val_years="2000-2005",
                 test_years="2006-2006"):
        super().__init__()
        self.labels = labels
        self.train_years = parse_years(train_years)
        self.val_years = parse_years(val_years)
        self.test_years = parse_years(test_years)
        self.all_years = list(set(self.train_years + \
            self.val_years + self.test_years))
        
        self.data_dic = np.load(year_data_path,
                                allow_pickle=True)['arr_0'][()]
        self.info_dic = np.load(wave_pr_path,
                                allow_pickle=True)['arr_0'][()]
        self.label_dic = {}

        for split in ["train", "val", "test"]:
            year = getattr(self, f"{split}_years")
            keylist = year_symbol_list(year, self.data_dic, 128)
            label_counts = label_wavepr_dataset(
                keylist, self.label_dic, self.info_dic)
            ds = DictListSampler(keylist,
                self.data_dic, self.label_dic, self.info_dic)
            ds.set_window(128 if split == "train" else 0)
            setattr(self, f"{split}_counts", label_counts)
            setattr(self, f"{split}_ds", ds)

    def __str__(self):
        s = f"=> Pointwise dataset\n"
        for split in ["train", "val", "test"]:
            counts = getattr(self, f"{split}_counts")
            N = sum(counts)
            s += f"=> {split} items {N}\n"
            for i in range(len(counts)):
                s += f"=> {self.labels[i]}:"
                s += f"{counts[i]}({int(counts[i]/N*100)}%) "
            s += "\n"
        return s

    def train_dataloader(self, batch_size=64, return_info=False):
        self.train_ds.return_info = return_info
        return DataLoader(self.train_ds,
            batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self, batch_size=1):
        self.val_ds.return_info = False
        return DataLoader(self.val_ds,
            batch_size=batch_size, shuffle=False)
    
    def test_dataloader(self, batch_size=1):
        self.test_ds.return_info = False
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