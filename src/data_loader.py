import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from pathlib import Path                        
from torch.utils.data import Dataset, DataLoader

base = Path(__file__).resolve().parent.parent / "data"

TRAIN_PATH = str(base / "5x127x127_training_with_morphology.hdf5")
VAL_PATH   = str(base / "5x127x127_validation_with_morphology.hdf5")
TEST_PATH  = str(base / "5x127x127_testing_with_morphology.hdf5")


MORPH_COLS = [
    "g_ellipticity", "g_half_light_radius", "g_isophotal_area", "g_major_axis", "g_minor_axis",
    "g_peak_surface_brightness", "g_petro_rad", "g_pos_angle", "g_sersic_index",
    "g_central_image_pop_5px_rad", "g_central_image_pop_10px_rad", "g_central_image_pop_15px_rad",
    "r_ellipticity", "r_half_light_radius", "r_isophotal_area", "r_major_axis", "r_minor_axis",
    "r_peak_surface_brightness", "r_petro_rad", "r_pos_angle", "r_sersic_index",
    "r_central_image_pop_5px_rad", "r_central_image_pop_10px_rad", "r_central_image_pop_15px_rad",
    "i_ellipticity", "i_half_light_radius", "i_isophotal_area", "i_major_axis", "i_minor_axis",
    "i_peak_surface_brightness", "i_petro_rad", "i_pos_angle", "i_sersic_index",
    "i_central_image_pop_5px_rad", "i_central_image_pop_10px_rad", "i_central_image_pop_15px_rad",
    "z_ellipticity", "z_half_light_radius", "z_isophotal_area", "z_major_axis", "z_minor_axis",
    "z_peak_surface_brightness", "z_petro_rad", "z_pos_angle", "z_sersic_index",
    "z_central_image_pop_5px_rad", "z_central_image_pop_10px_rad", "z_central_image_pop_15px_rad",
    "y_ellipticity", "y_half_light_radius", "y_isophotal_area", "y_major_axis", "y_minor_axis",
    "y_peak_surface_brightness", "y_petro_rad", "y_pos_angle", "y_sersic_index",
    "y_central_image_pop_5px_rad", "y_central_image_pop_10px_rad", "y_central_image_pop_15px_rad",
]


# Computed with the following script: scripts/compute_band_stats.py
BAND_MEAN = torch.tensor(
    [0.07781, 0.15729, 0.23103, 0.30393, 0.36941], dtype=torch.float32
).view(-1, 1, 1)

BAND_STD = torch.tensor(
    [0.82795, 1.44230, 1.78690, 2.59873, 3.14001], dtype=torch.float32
).view(-1, 1, 1)


_BANDS = ["g", "r", "i", "z", "y"]
MAG_COLS = (
    [f"{b}_cmodel_mag"      for b in _BANDS] +
    [f"{b}_cmodel_magsigma" for b in _BANDS]
)
TABULAR_COLS = MAG_COLS + MORPH_COLS  

EXCLUDE = {
    'image',
    'specz_redshift',
    'specz_redshift_err',
    'specz_flag_homogeneous',
}


class GalaxyDataset(Dataset):
    """
    GalaxiesML Dataset - loads 5-band images and spectroscopic redshift labels.
    """

    def __init__(
        self,
        path: str,
        augment: bool = False,
        mean: torch.Tensor = BAND_MEAN,
        std:  torch.Tensor = BAND_STD,
        resize= None,
    ) -> None:
        self.path = path
        self.augment = augment
        self.resize = resize
        self.mean = mean
        self.std  = std

        with h5py.File(path, "r") as f:
            flags = f["specz_flag_homogeneous"][:].astype(bool)
            self.indices = np.where(flags)[0]

        print(f"Loaded {Path(path).name}: {len(self.indices):,} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        with h5py.File(self.path, "r") as f:
            img = f["image"][real_idx].astype(np.float32)  
            y = float(f["specz_redshift"][real_idx])

        x = torch.from_numpy(img)

        if self.resize:                                    
            x = TF.resize(x, [self.resize, self.resize],
                          interpolation=TF.InterpolationMode.BICUBIC,
                          antialias=True)
            
        x = (x - self.mean) / self.std                  

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = TF.hflip(x)
            if torch.rand(1).item() > 0.5:
                x = TF.vflip(x)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=[-2, -1])       

        return x, torch.tensor(y, dtype=torch.float32)


class FusionGalaxyDataset(Dataset):
    """
    GalaxiesML Dataset for fusion models — returns (image, morph, label).
    """

    def __init__(
        self,
        path: str,
        df,
        img_mean: torch.Tensor,
        img_std: torch.Tensor,
        morph_mean: torch.Tensor,
        morph_std: torch.Tensor,
        augment: bool = False,
    ) -> None:
        self.path = path
        self.augment = augment
        self.img_mean = img_mean
        self.img_std = img_std

        with h5py.File(path, "r") as f:
            flags = f["specz_flag_homogeneous"][:].astype(bool)
        self.indices = np.where(flags)[0]

        morph_raw = df[MORPH_COLS].values.astype(np.float32)
        morph_raw = np.where(np.isfinite(morph_raw), morph_raw, 0.0)
        self.morph = torch.tensor(
            (morph_raw - morph_mean.numpy()) / morph_std.numpy(), dtype=torch.float32
        )
        self.labels = torch.tensor(df["specz_redshift"].values, dtype=torch.float32)

        print(f"Loaded {Path(path).name}: {len(self.indices):,} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        with h5py.File(self.path, "r") as f:
            img = f["image"][self.indices[idx]].astype(np.float32)
        x = (torch.from_numpy(img) - self.img_mean) / self.img_std
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = TF.hflip(x)
            if torch.rand(1).item() > 0.5:
                x = TF.vflip(x)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=[-2, -1])
        return x, self.morph[idx], self.labels[idx]
    

class PhysFusionGalaxyDataset(Dataset):
    """
    Dataset for the physically-informed fusion model - returns (image, tabular, label).
    """
    def __init__(
        self,
        path: str,
        tabular_cols=None,
        augment: bool = False,
        img_mean: torch.Tensor = BAND_MEAN,
        img_std: torch.Tensor = BAND_STD,
        tab_mean=None,
        tab_std=None,
    ) -> None:
        if tabular_cols is None:
            tabular_cols = TABULAR_COLS
        self.path = path
        self.tabular_cols = tabular_cols
        self.augment = augment
        self.img_mean = img_mean
        self.img_std = img_std

        with h5py.File(path, "r") as f:
            flags = f["specz_flag_homogeneous"][:].astype(bool)
        self.indices = np.where(flags)[0]

        with h5py.File(path, "r") as f:
            self.labels = f["specz_redshift"][self.indices].astype(np.float32)
            self.tabular = np.stack(
                [f[c][self.indices].astype(np.float32) for c in tabular_cols],
                axis=1,
            )

        if tab_mean is None or tab_std is None:
            raw = self._tabular_with_colors()
            self.tab_mean = raw.mean(axis=0).astype(np.float32)
            self.tab_std  = (raw.std(axis=0) + 1e-8).astype(np.float32)
        else:
            self.tab_mean = tab_mean
            self.tab_std = tab_std

        self._file = h5py.File(path, "r")
        print(f"Loaded {Path(path).name}: {len(self.indices):,} samples")

    def _tabular_with_colors(self):
        tab = self.tabular
        g, r, i, z, y = tab[:, 0], tab[:, 1], tab[:, 2], tab[:, 3], tab[:, 4]
        colors = np.stack([g-r, r-i, i-z, z-y, g-i, g-z, r-z, r-y, g-y], axis=1)
        return np.concatenate([tab, colors], axis=1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img = self._file["image"][self.indices[idx]].astype(np.float32)
        label = float(self.labels[idx])

        tab = self.tabular[idx].copy()
        g, r, i, z, y = tab[0], tab[1], tab[2], tab[3], tab[4]
        colors = np.array([g-r, r-i, i-z, z-y, g-i, g-z, r-z, r-y, g-y], dtype=np.float32)
        tab = np.concatenate([tab, colors])
        tab = (tab - self.tab_mean) / self.tab_std
        tab = np.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(img)
        x = (x - self.img_mean) / self.img_std
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = TF.hflip(x)
            if torch.rand(1).item() > 0.5:
                x = TF.vflip(x)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=[-2, -1])

        return (
            x,
            torch.tensor(tab,   dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

    def __del__(self):
        try:
            if hasattr(self, "_file") and self._file.id.valid:
                self._file.close()
        except Exception:
            pass

    
def get_dataloaders(
    train_path: str,
    val_path:   str,
    test_path:  str,
    batch_size:  int = 32,
    num_workers: int = 0,
    resize: int = None, 
) -> tuple:
    """
    Build all three DataLoaders in one call.
    """
    train_ds = GalaxyDataset(train_path, augment=True, resize=resize)
    val_ds = GalaxyDataset(val_path, augment=False, resize=resize)
    test_ds = GalaxyDataset(test_path, augment=False, resize=resize)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader


def get_fusion_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    df_train,
    df_val,
    df_test,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    """
    Build fusion DataLoaders that return (image, morph, label) batches.
    """
    morph_raw = df_train[MORPH_COLS].values.astype(np.float32)
    morph_raw = np.where(np.isfinite(morph_raw), morph_raw, 0.0)
    morph_mean = torch.tensor(morph_raw.mean(axis=0), dtype=torch.float32)
    morph_std = torch.tensor(np.maximum(morph_raw.std(axis=0), 1e-8), dtype=torch.float32)

    train_ds = FusionGalaxyDataset(train_path, df_train, BAND_MEAN, BAND_STD, morph_mean, morph_std, augment=True)
    val_ds = FusionGalaxyDataset(val_path, df_val, BAND_MEAN, BAND_STD, morph_mean, morph_std, augment=False)
    test_ds = FusionGalaxyDataset(test_path, df_test, BAND_MEAN, BAND_STD, morph_mean, morph_std, augment=False)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, morph_mean, morph_std


def get_phys_fusion_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tabular_cols=None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple:
    """
    Build DataLoaders for the physically-informed fusion model.
    """
    if tabular_cols is None:
        tabular_cols = TABULAR_COLS

    train_ds = PhysFusionGalaxyDataset(train_path, tabular_cols, augment=True)
    val_ds = PhysFusionGalaxyDataset(val_path, tabular_cols, augment=False,
                                        tab_mean=train_ds.tab_mean, tab_std=train_ds.tab_std)
    test_ds = PhysFusionGalaxyDataset(test_path, tabular_cols, augment=False,
                                        tab_mean=train_ds.tab_mean, tab_std=train_ds.tab_std)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, train_ds.tab_mean, train_ds.tab_std


def load_split(path: str) -> pd.DataFrame:
    """
    Load tabular photometric features from a GalaxiesML HDF5 split into a DataFrame.
    """
    with h5py.File(path, "r") as f:
        data = {}
        for key in sorted(f.keys()):
            if key in EXCLUDE:
                continue
            arr = f[key][:]
            if arr.ndim == 1:
                data[key] = arr
            elif arr.ndim == 2:
                for i in range(arr.shape[1]):
                    data[f"{key}_{i}"] = arr[:, i]

        label = f["specz_redshift"][:].astype(np.float64)
        flag = f["specz_flag_homogeneous"][:].astype(bool)

    df = pd.DataFrame(data)
    df["specz_redshift"] = label
    df["specz_flag_homogeneous"] = flag

    df = df[df["specz_flag_homogeneous"]].reset_index(drop=True)
    df = df.drop(columns=["specz_flag_homogeneous"])

    print(f"Loaded {Path(path).name}: {len(df):,} samples, {df.shape[1]-1} features")
    return df

