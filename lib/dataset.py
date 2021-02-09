import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import numpy as np
import glob

from lib.utils import *


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