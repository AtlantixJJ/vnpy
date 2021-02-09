import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import numpy as np
import glob


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
                #print(f"=> {k1} {k2} : {dic[k1][k2].shape}")
                data.append(dic[k1][k2])
    if len(data) == 0:
        return None
    x = torch.from_numpy(np.concatenate(data))
    if torch.isnan(x).sum():
        return None
    return x


def balance_class(xs, ys, max_dev=1.1):
    """Balancing each category.
    Only re-sample the most imbalanced class.
    """
    lens = np.array([y.shape[0] for y in ys])
    ratios = lens / lens.sum().astype("float32")
    indice = np.argsort(lens)
    maxi2, maxi1 = indice[-2:]
    if ratios[maxi1] > max_dev * ratios[maxi2]:
        sample_rate = max_dev * lens[maxi2] / lens[maxi1]
        N_total = xs[maxi1].shape[0]
        length = int(sample_rate * N_total)
        new_ratio = length / (lens.sum() - N_total + length)
        print(f"=> Balance resample class {maxi1} from {ratios[maxi1]*100:.2f}% to {new_ratio*100:.2f}%")
        rng = np.random.RandomState(1)
        indice = rng.choice(np.arange(0, N_total),
            size=(length,), replace=False)
        xs[-1] = xs[-1][indice]
        ys[-1] = ys[-1][indice]
    return xs, ys



def load_year(res, data_dir="data/buy_point", years=[2000]):
    files = glob.glob(f"{data_dir}/share*")
    N = len(files)

    def assign(data_key, symbol, year):
        res[data_key][symbol][year] = dic[data_key][symbol][year]

    for i in range(1, N + 1):
        fpath = f"{data_dir}/share_{i}.npy"
        print(f"=> Loading {fpath}")
        dic = np.load(fpath, allow_pickle=True)[()]
        for data_key in dic.keys():
            if data_key not in res:
                res[data_key] = {}
            for symbol in dic[data_key]:
                if symbol not in res[data_key]:
                    res[data_key][symbol] = {}
                for year in years:
                    year = str(year)
                    if year not in dic[data_key][symbol]:
                        continue
                    assign(data_key, symbol, year)


class BuyPoint(pl.LightningDataModule):
    def __init__(self,
                 data_dir="data/buy_point"):
        """Create training and validation split from raw data.
        Args:
            dic : dic[symbol][year]
        """
        super().__init__()
        self.dic = {}
        load_year(self.dic, data_dir, years=range(2000, 2005))
        self.label_keys = list(self.dic.keys())
        k = self.label_keys[0]
        self.all_symbols = np.array(list(self.dic[k].keys()))
        N_symbols = self.all_symbols.shape[0]
        self.N_symbols = N_symbols

        self.train_symbols, self.val_symbols, self.test_symbols = \
            split(self.all_symbols, 0.6, 0.2)
        
        self.train_years = ["2000", "2001", "2002"]
        self.val_years = ["2003"]
        self.test_years = ["2004"]
        self.create_datasets()

    def create_datasets(self):
        for split in ["train", "val", "test"]:
            x, y = [], []
            for i, k in enumerate(self.label_keys):
                val = tensor_from_dict2d(
                    dic=self.dic[k],
                    keys1=getattr(self, f"{split}_symbols"),
                    keys2=getattr(self, f"{split}_years"))
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