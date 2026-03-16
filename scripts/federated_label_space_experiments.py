"""
Federated learning experiments under partial label-space overlap.

Implements four strategies:
- Union
- Naive Union
- Unknown
- Isolation

Code accompanying the paper:
"Federated Learning under Partial Label-Space Overlap"
"""

import os
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*start_simulation\\(\\) is deprecated.*")
warnings.filterwarnings("ignore", message=".*client_fn.*expects a signature.*Context.*")
warnings.filterwarnings("ignore", message=".*Failed to establish connection to the metrics exporter agent.*")
warnings.filterwarnings("ignore", message=".*Tensor\\.pin_memory\\(\\) is deprecated.*")
warnings.filterwarnings("ignore", message=".*Tensor\\.is_pinned\\(\\) is deprecated.*")

# ============================================================
# IMPORTS
# ============================================================
import datetime
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
import torchvision.models as torch_models
import torchxrayvision as xrv

import flwr
from flwr.client import NumPyClient, Client
from flwr.server.strategy import FedProx
from flwr.simulation import start_simulation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# ============================================================
# REPRODUCIBILITY
# ============================================================
import random
# SEED = 123
# SEED = 456
SEED = 42
# SEED = int(os.environ.get("SEED", "123"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SPLIT_SEED_BASE = 84 + SEED

# ============================================================
# DATALOADER 
# Helper functions to create deterministic PyTorch dataloaders.
# ============================================================
def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dl_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def make_dataloader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 2):
    g = make_dl_generator(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False,
    )

# ============================================================
# GENERAL CONFIGURATION
# Global training parameters and system settings.
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")

NUM_CLIENTS = 4
BATCH_SIZE = 32
NUM_ROUNDS = 60
# NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "60"))

GLOBAL_EPOCHS = 1
PRIVATE_EPOCHS = 2

GLOBAL_CLASSES = ["COVID", "NORMAL", "PNEUMONIA"]
GLOBAL_SET = set(GLOBAL_CLASSES)

# ============================================================
# EXPERIMENTAL MODES 
# "isolation"   : train only on shared global classes (private head handled separately)
# "unknown"     : map non-shared classes to UNKNOWN
# "union"       : train on the union of all classes with label alignment by name
# "union_naive" : train on the union without cross-client label alignment
# ============================================================
EXPERIMENT_MODE = "union"  
# EXPERIMENT_MODE = os.environ.get("EXPERIMENT_MODE", "isolation")
UNKNOWN_LABEL_NAME = "UNKNOWN"

# print(f"[CONFIG OVERRIDE] EXPERIMENT_MODE={EXPERIMENT_MODE} SEED={SEED} NUM_ROUNDS={NUM_ROUNDS}")
# ============================================================
# HYPERPARAMETERS
# ============================================================
PROXIMAL_MU = 0.02
CB_BETA = 0.99
LR_GLOBAL = 3e-4
LABEL_SMOOTHING = 0.0

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("experiments", f"run_{timestamp}_{EXPERIMENT_MODE}")
os.makedirs(results_dir, exist_ok=True)

global_dir = os.path.join(results_dir, "global")
private_dir = os.path.join(results_dir, "private")
os.makedirs(global_dir, exist_ok=True)
os.makedirs(private_dir, exist_ok=True)

per_round_global = os.path.join(global_dir, "per_round")
final_eval_global = os.path.join(global_dir, "final_evaluation")
os.makedirs(per_round_global, exist_ok=True)
os.makedirs(final_eval_global, exist_ok=True)

per_round_private = os.path.join(private_dir, "per_round")
final_eval_private = os.path.join(private_dir, "final_evaluation")
os.makedirs(per_round_private, exist_ok=True)
os.makedirs(final_eval_private, exist_ok=True)

server_metrics_path = os.path.join(global_dir, "server_per_round_metrics.csv")
server_fit_metrics_path = os.path.join(global_dir, "server_fit_metrics.csv")

CLIENT_STATE_DIR = os.path.join(results_dir, "client_state")
os.makedirs(CLIENT_STATE_DIR, exist_ok=True)

def private_ckpt_path(partition_id: int) -> str:
    return os.path.join(CLIENT_STATE_DIR, f"client_{partition_id}_private_classifier.pt")

CLIENT_DATASETS = {}

# ============================================================
# CLIENT DATA DIRECTORIES
# ============================================================
CLIENT_DATA_DIRS = [
    "paciente1",
    "paciente2_completo",
    "paciente3_completo",
    "paciente4_completo"
]

# ============================================================
# GLOBAL AND PRIVATE LABEL SPACE DEFINITION
# Identify shared (global) and client-specific (private) classes.
# ============================================================
def _get_base_folder_for_scan(d: str) -> str:
    if os.path.isdir(os.path.join(d, "train")):
        return os.path.join(d, "train")
    return d

def scan_all_classes(client_dirs):
    all_classes = set()
    for d in client_dirs:
        base = _get_base_folder_for_scan(d)
        ds = ImageFolder(base)
        all_classes |= set(ds.classes)
    return all_classes

def scan_max_local_num_classes(client_dirs) -> int:
    mx = 0
    for d in client_dirs:
        base = _get_base_folder_for_scan(d)
        ds = ImageFolder(base)
        mx = max(mx, len(ds.classes))
    return mx

ALL_CLASSES = scan_all_classes(CLIENT_DATA_DIRS)
PRIVATE_UNION = sorted(list(ALL_CLASSES - GLOBAL_SET))
MAX_LOCAL_CLASSES = scan_max_local_num_classes(CLIENT_DATA_DIRS)

if EXPERIMENT_MODE == "union":
    EXPERIMENT_CLASSES = GLOBAL_CLASSES + PRIVATE_UNION
elif EXPERIMENT_MODE == "unknown":
    EXPERIMENT_CLASSES = GLOBAL_CLASSES + [UNKNOWN_LABEL_NAME]
elif EXPERIMENT_MODE == "isolation":
    EXPERIMENT_CLASSES = GLOBAL_CLASSES
elif EXPERIMENT_MODE == "union_naive":
    EXPERIMENT_CLASSES = [f"LOCAL_{i}" for i in range(MAX_LOCAL_CLASSES)]
else:
    raise ValueError("EXPERIMENT_MODE inválido: usa 'isolation'|'unknown'|'union'|'union_naive'")

EXPERIMENT_CLASS_TO_IDX = {c: i for i, c in enumerate(EXPERIMENT_CLASSES)}
IDX_TO_EXPERIMENT_CLASS = {i: c for c, i in EXPERIMENT_CLASS_TO_IDX.items()}
NUM_EXPERIMENT_CLASSES = len(EXPERIMENT_CLASSES)

# Global label mapping (not used in union_naive mode)
if EXPERIMENT_MODE != "union_naive":
    GLOBAL_IDX_IN_EXPERIMENT = [EXPERIMENT_CLASS_TO_IDX[c] for c in GLOBAL_CLASSES]
    EXP_TO_GLOBAL_MAP = {exp_i: gi for gi, exp_i in enumerate(GLOBAL_IDX_IN_EXPERIMENT)}
else:
    GLOBAL_IDX_IN_EXPERIMENT = None
    EXP_TO_GLOBAL_MAP = None

# Private class indices in union mode (used for private-only metrics)
PRIVATE_IDX_IN_EXPERIMENT = None
if EXPERIMENT_MODE == "union":
    PRIVATE_IDX_IN_EXPERIMENT = [EXPERIMENT_CLASS_TO_IDX[c] for c in PRIVATE_UNION]

UNKNOWN_IDX = None
if UNKNOWN_LABEL_NAME in EXPERIMENT_CLASS_TO_IDX:
    UNKNOWN_IDX = EXPERIMENT_CLASS_TO_IDX[UNKNOWN_LABEL_NAME]

print(f"[MODE={EXPERIMENT_MODE}] NUM_EXPERIMENT_CLASSES={NUM_EXPERIMENT_CLASSES}")
print(f"[MODE={EXPERIMENT_MODE}] EXPERIMENT_CLASSES={EXPERIMENT_CLASSES}")

# ============================================================
# DATASETS
# ============================================================
class RemappedDataset(Dataset):
    """
    Remap local labels to the experiment label space.

    Modes:
    - union: align labels by class name
    - unknown: map non-global classes to UNKNOWN
    - isolation: discard non-global classes
    - union_naive: keep local labels unchanged and store a
    local-to-global mapping for global-only evaluation
    """
    def __init__(self, base_dataset, mode, global_set, class_to_idx, unknown_name=UNKNOWN_LABEL_NAME):
        self.samples, self.targets = [], []
        self.transform = base_dataset.transform
        self.loader = base_dataset.loader

        self.base_classes = list(getattr(base_dataset, "classes", []))
        self.local_to_global = {}
        if mode == "union_naive":
            for gi, gname in enumerate(GLOBAL_CLASSES):
                if gname in self.base_classes:
                    li = self.base_classes.index(gname)
                    self.local_to_global[int(li)] = int(gi)

        for path, local_label in base_dataset.samples:
            class_name = base_dataset.classes[local_label]

            if mode == "union":
                if class_name in class_to_idx:
                    self.samples.append(path)
                    self.targets.append(class_to_idx[class_name])

            elif mode == "unknown":
                mapped = class_to_idx[class_name] if class_name in global_set else class_to_idx[unknown_name]
                self.samples.append(path)
                self.targets.append(mapped)

            elif mode == "isolation":
                if class_name in global_set:
                    self.samples.append(path)
                    self.targets.append(class_to_idx[class_name])

            elif mode == "union_naive":
                self.samples.append(path)
                self.targets.append(int(local_label))

            else:
                raise ValueError("mode inválido")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.loader(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


class PrivateDataset(Dataset):
    """Filter private classes and remap them to consecutive local indices using a deterministic ordering (alphabetical)."""
    def __init__(self, base_dataset, private_classes):
        self.samples, self.targets = [], []
        self.transform = base_dataset.transform
        self.loader = base_dataset.loader

        private_classes_sorted = sorted(list(private_classes))
        self.class_to_idx = {c: i for i, c in enumerate(private_classes_sorted)}

        for path, local_label in base_dataset.samples:
            class_name = base_dataset.classes[local_label]
            if class_name in self.class_to_idx:
                self.samples.append(path)
                self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.loader(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def compute_dataset_stats(dataset: Dataset, seed: int):
    loader = make_dataloader(dataset, batch_size=64, shuffle=False, seed=seed, num_workers=0)
    mean, std, n_batches = 0.0, 0.0, 0
    for imgs, _ in loader:
        mean += imgs.mean(dim=[0, 2, 3])
        std += imgs.std(dim=[0, 2, 3])
        n_batches += 1
    mean /= max(n_batches, 1)
    std /= max(n_batches, 1)
    return mean.tolist(), std.tolist()

def split_indices(n: int, seed: int, frac_test_final=0.20, frac_val_from_rest=0.20):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_test = int(frac_test_final * n)
    test_final_idx = perm[:n_test]
    rest = perm[n_test:]
    n_val = int(frac_val_from_rest * len(rest))
    val_idx = rest[:n_val]
    train_idx = rest[n_val:]
    return train_idx, val_idx, test_final_idx

def split_indices_2(n: int, seed: int, frac_val=0.20):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(frac_val * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx

def load_datasets(partition_id, dataset_path, batch_size=BATCH_SIZE):
    has_train_test_dirs = os.path.isdir(os.path.join(dataset_path, "train")) and \
                          os.path.isdir(os.path.join(dataset_path, "test"))

    if has_train_test_dirs:
        train_base = ImageFolder(os.path.join(dataset_path, "train"))
        test_base = ImageFolder(os.path.join(dataset_path, "test"))
    else:
        train_base = ImageFolder(dataset_path)
        test_base = None

    local_classes = set(train_base.classes)
    private_classes = local_classes - GLOBAL_SET

    temp_transform = Compose([Grayscale(1), Resize((224, 224)), ToTensor()])

    dl_seed_global = int(SPLIT_SEED_BASE + 10_000 * partition_id)
    dl_seed_private = int(SPLIT_SEED_BASE + 20_000 * partition_id)

    # ---------------- GLOBAL ----------------
    global_train = global_val = global_test_final = None

    global_train_dataset = RemappedDataset(
        deepcopy(train_base),
        mode=EXPERIMENT_MODE,
        global_set=GLOBAL_SET,
        class_to_idx=EXPERIMENT_CLASS_TO_IDX,
        unknown_name=UNKNOWN_LABEL_NAME
    )
    global_train_dataset.transform = temp_transform

    if has_train_test_dirs:
        global_test_final_dataset = RemappedDataset(
            deepcopy(test_base),
            mode=EXPERIMENT_MODE,
            global_set=GLOBAL_SET,
            class_to_idx=EXPERIMENT_CLASS_TO_IDX,
            unknown_name=UNKNOWN_LABEL_NAME
        )
        global_test_final_dataset.transform = temp_transform
    else:
        global_test_final_dataset = None

    if len(global_train_dataset) > 0:
        n = len(global_train_dataset)

        if has_train_test_dirs:
            train_idx, val_idx = split_indices_2(n, seed=SPLIT_SEED_BASE + partition_id, frac_val=0.20)

            train_subset_for_stats = Subset(global_train_dataset, train_idx)
            mean, std = compute_dataset_stats(train_subset_for_stats, seed=dl_seed_global + 100)
            print(f"[Client {partition_id}] GLOBAL stats ({EXPERIMENT_MODE}) - mean: {mean}, std: {std}")

            normalize_transform = Compose([
                Grayscale(1), Resize((224, 224)), ToTensor(),
                Normalize(tuple(mean), tuple(std))
            ])
            global_train_dataset.transform = normalize_transform
            global_test_final_dataset.transform = normalize_transform

            global_train = make_dataloader(Subset(global_train_dataset, train_idx), batch_size, True, dl_seed_global + 1, num_workers=0)
            global_val   = make_dataloader(Subset(global_train_dataset, val_idx),   batch_size, False, dl_seed_global + 2, num_workers=0)
            global_test_final = make_dataloader(global_test_final_dataset, batch_size, False, dl_seed_global + 3, num_workers=0)
        else:
            train_idx, val_idx, test_final_idx = split_indices(
                n, seed=SPLIT_SEED_BASE + partition_id, frac_test_final=0.20, frac_val_from_rest=0.20
            )

            train_subset_for_stats = Subset(global_train_dataset, train_idx)
            mean, std = compute_dataset_stats(train_subset_for_stats, seed=dl_seed_global + 100)
            print(f"[Client {partition_id}] GLOBAL stats ({EXPERIMENT_MODE}) - mean: {mean}, std: {std}")

            normalize_transform = Compose([
                Grayscale(1), Resize((224, 224)), ToTensor(),
                Normalize(tuple(mean), tuple(std))
            ])
            global_train_dataset.transform = normalize_transform

            global_train = make_dataloader(Subset(global_train_dataset, train_idx), batch_size, True,  dl_seed_global + 1, num_workers=0)
            global_val   = make_dataloader(Subset(global_train_dataset, val_idx),   batch_size, False, dl_seed_global + 2, num_workers=0)
            global_test_final = make_dataloader(Subset(global_train_dataset, test_final_idx), batch_size, False, dl_seed_global + 3, num_workers=0)

    # ---------------- PRIVATE (isolation only) ----------------
    private_train = private_val = private_test_final = None

    if (EXPERIMENT_MODE == "isolation") and private_classes:
        private_train_dataset = PrivateDataset(deepcopy(train_base), private_classes)
        private_train_dataset.transform = temp_transform

        if has_train_test_dirs:
            private_test_final_dataset = PrivateDataset(deepcopy(test_base), private_classes)
            private_test_final_dataset.transform = temp_transform
        else:
            private_test_final_dataset = None

        if len(private_train_dataset) > 0:
            n = len(private_train_dataset)

            if has_train_test_dirs:
                train_idx, val_idx = split_indices_2(n, seed=SPLIT_SEED_BASE + partition_id, frac_val=0.20)

                train_subset_for_stats = Subset(private_train_dataset, train_idx)
                mean, std = compute_dataset_stats(train_subset_for_stats, seed=dl_seed_private + 100)
                print(f"[Client {partition_id}] PRIVATE stats - mean: {mean}, std: {std}")

                normalize_transform = Compose([
                    Grayscale(1), Resize((224, 224)), ToTensor(),
                    Normalize(tuple(mean), tuple(std))
                ])
                private_train_dataset.transform = normalize_transform
                private_test_final_dataset.transform = normalize_transform

                private_train = make_dataloader(Subset(private_train_dataset, train_idx), batch_size, True,  dl_seed_private + 1, num_workers=0)
                private_val   = make_dataloader(Subset(private_train_dataset, val_idx),   batch_size, False, dl_seed_private + 2, num_workers=0)
                private_test_final = make_dataloader(private_test_final_dataset, batch_size, False, dl_seed_private + 3, num_workers=0)
            else:
                train_idx, val_idx, test_final_idx = split_indices(
                    n, seed=SPLIT_SEED_BASE + partition_id, frac_test_final=0.20, frac_val_from_rest=0.20
                )

                train_subset_for_stats = Subset(private_train_dataset, train_idx)
                mean, std = compute_dataset_stats(train_subset_for_stats, seed=dl_seed_private + 100)
                print(f"[Client {partition_id}] PRIVATE stats - mean: {mean}, std: {std}")

                normalize_transform = Compose([
                    Grayscale(1), Resize((224, 224)), ToTensor(),
                    Normalize(tuple(mean), tuple(std))
                ])
                private_train_dataset.transform = normalize_transform

                private_train = make_dataloader(Subset(private_train_dataset, train_idx), batch_size, True,  dl_seed_private + 1, num_workers=0)
                private_val   = make_dataloader(Subset(private_train_dataset, val_idx),   batch_size, False, dl_seed_private + 2, num_workers=0)
                private_test_final = make_dataloader(Subset(private_train_dataset, test_final_idx), batch_size, False, dl_seed_private + 3, num_workers=0)

    CLIENT_DATASETS[partition_id] = {
        "global_train": global_train,
        "global_val": global_val,
        "global_test_final": global_test_final,
        "private_train": private_train,
        "private_val": private_val,
        "private_test_final": private_test_final,
    }
    return global_train, global_val, global_test_final, private_train, private_val, private_test_final

# ============================================================
# GLOBAL MODEL
# ============================================================
class GlobalModel(nn.Module):
    def __init__(self, num_classes: int, load_pretrained: bool = True):
        super().__init__()
        self.backbone = torch_models.densenet121()
        self.backbone.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        if load_pretrained:
            xrv_model = xrv.models.DenseNet(weights="densenet121-res224-chex")
            state_dict = {k: v for k, v in xrv_model.state_dict().items() if "classifier" not in k}
            self.backbone.load_state_dict(state_dict, strict=False)

        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def get_parameters(model: nn.Module):
    return [val.detach().cpu().numpy() for val in model.state_dict().values()]

def set_parameters(model: nn.Module, parameters):
    state_dict = model.state_dict()
    for (k, old), v in zip(state_dict.items(), parameters):
        t = torch.from_numpy(v).to(device=old.device, dtype=old.dtype)
        state_dict[k] = t
    model.load_state_dict(state_dict, strict=True)

# ============================================================
# HELPER FUNCTIONS
# Utilities for mapping labels, extracting global predictions,
# and computing evaluation subsets.
# ============================================================
def _get_global_mapping_from_loader(loader: DataLoader):
    """
    Returns:
    - cols_global_in_head: indices of logits corresponding to the global classes
    [COVID, NORMAL, PNEUMONIA]
    - local_label_to_global: mapping from dataset labels to global class indices (0..2),
    used to identify global examples
    """
    if EXPERIMENT_MODE != "union_naive":
        cols = list(GLOBAL_IDX_IN_EXPERIMENT)
        local_to_global = dict(EXP_TO_GLOBAL_MAP)
        return cols, local_to_global

    ds = getattr(loader, "dataset", None)
    if isinstance(ds, Subset):
        ds = ds.dataset

    local_to_global = getattr(ds, "local_to_global", {}) or {}

    cols = [None, None, None]
    for local_idx, gidx in local_to_global.items():
        if 0 <= int(gidx) <= 2:
            cols[int(gidx)] = int(local_idx)

    if any(c is None for c in cols):
        return None, local_to_global

    return cols, local_to_global

def _filter_to_global_only_from_batch(labels: torch.Tensor, local_label_to_global: dict):
    labels_np = labels.detach().cpu().numpy().astype(int)
    keep = np.array([int(l) in local_label_to_global for l in labels_np], dtype=bool)
    if keep.sum() == 0:
        return keep, np.array([], dtype=int)
    y_true_g = np.array([local_label_to_global[int(l)] for l in labels_np[keep]], dtype=int)
    return keep, y_true_g

def _global_restricted_preds_and_probs_modeaware(logits: torch.Tensor, cols_global_in_head):
    logits_g = logits[:, cols_global_in_head]
    probs_g = torch.softmax(logits_g, dim=1)
    preds_g = torch.argmax(logits_g, dim=1)
    return preds_g, probs_g, logits_g

def _unknown_rate_on_global_examples(y_true_exp_np: np.ndarray, y_pred_exp_np: np.ndarray, local_label_to_global: dict):
    if EXPERIMENT_MODE != "unknown":
        return np.nan
    if UNKNOWN_IDX is None:
        return np.nan
    keep = np.array([int(t) in local_label_to_global for t in y_true_exp_np], dtype=bool)
    if keep.sum() == 0:
        return np.nan
    return float((y_pred_exp_np[keep] == UNKNOWN_IDX).mean())


# ============================================================
# GLOBAL TRAINING (FedProx)
# ============================================================
def train_global(global_model=None, global_loader=None, global_params=None, proximal_mu=0.0, epochs=1):
    results = {}
    if global_model is None or global_loader is None:
        return results

    ds = getattr(global_loader, "dataset", None)
    targets = None
    try:
        if isinstance(ds, Subset):
            base = ds.dataset
            idxs = np.asarray(ds.indices, dtype=int)
            if hasattr(base, "targets"):
                base_targets = np.asarray(base.targets, dtype=int)
                targets = base_targets[idxs]
        else:
            if hasattr(ds, "targets"):
                targets = np.asarray(ds.targets, dtype=int)
    except Exception:
        targets = None

    if targets is None or targets.size == 0:
        results["global"] = {"exp_train_loss": np.nan, "exp_train_acc": np.nan}
        return results

    class_counts = np.bincount(targets.astype(int), minlength=NUM_EXPERIMENT_CLASSES)

    beta = CB_BETA
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights[class_counts == 0] = 0.0

    nz = weights > 0
    if np.any(nz):
        weights[nz] = weights[nz] / (weights[nz].mean() + 1e-12)

    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=LR_GLOBAL)

    global_model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for _ in range(int(epochs)):
        for imgs, labels in global_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = global_model(imgs)
            loss = criterion(outputs, labels)

            if global_params is not None and len(global_params) > 0:
                prox = 0.0
                for w, wg in zip(global_model.parameters(), global_params):
                    prox = prox + torch.sum((w - wg.to(DEVICE)) ** 2)
                loss = loss + (float(proximal_mu) / 2.0) * prox

            loss.backward()
            optimizer.step()

            bs = int(labels.size(0))
            total_loss += float(loss.item()) * bs
            total_correct += int((outputs.argmax(1) == labels).sum().item())
            total_samples += bs

    results["global"] = {
        "exp_train_loss": total_loss / max(total_samples, 1),
        "exp_train_acc": total_correct / max(total_samples, 1),
    }
    return results

# ============================================================
# GLOBAL EVALUATION
# Computes:
# - experiment-level metrics (full label space)
# - global-only metrics (restricted to shared classes)
# - private-only metrics in union mode
# ============================================================
def eval_global_basic_plus(model: nn.Module, loader: DataLoader):
    model.eval()
    criterion_full = nn.CrossEntropyLoss()

    cols_global_in_head, local_to_global = _get_global_mapping_from_loader(loader)

    total_loss_full, total_full, correct_full = 0.0, 0, 0
    y_true_exp, y_pred_exp = [], []
    y_true_g_all, y_pred_g_all = [], []
    global_loss_sum, global_count = 0.0, 0

    # union private-only accumulators
    priv_loss_sum, priv_count = 0.0, 0
    priv_correct = 0
    priv_to_global_hits = 0

    with torch.no_grad():
        for imgs, labels_exp in loader:
            imgs, labels_exp = imgs.to(DEVICE), labels_exp.to(DEVICE)
            logits = model(imgs)

            # FULL exp
            loss_full = criterion_full(logits, labels_exp)
            total_loss_full += loss_full.item() * labels_exp.size(0)
            preds_full = torch.argmax(logits, dim=1)
            correct_full += (preds_full == labels_exp).sum().item()
            total_full += labels_exp.size(0)

            y_true_exp.extend(labels_exp.detach().cpu().numpy())
            y_pred_exp.extend(preds_full.detach().cpu().numpy())

            # GLOBAL-only (examples + restricted head)
            keep, y_true_g_np = _filter_to_global_only_from_batch(labels_exp, local_to_global)
            if np.any(keep) and (cols_global_in_head is not None):
                keep_t = torch.as_tensor(keep, device=labels_exp.device, dtype=torch.bool)
                logits_kept = logits[keep_t]
                preds_g_t, _, logits_g = _global_restricted_preds_and_probs_modeaware(logits_kept, cols_global_in_head)

                y_true_g_t = torch.tensor(y_true_g_np, device=labels_exp.device, dtype=torch.long)
                loss_g = F.cross_entropy(logits_g, y_true_g_t, reduction="sum")
                global_loss_sum += loss_g.item()
                global_count += int(y_true_g_t.numel())

                y_true_g_all.extend(y_true_g_np.tolist())
                y_pred_g_all.extend(preds_g_t.detach().cpu().numpy().tolist())

            # UNION private-only
            if EXPERIMENT_MODE == "union" and PRIVATE_IDX_IN_EXPERIMENT is not None:
                y_np = labels_exp.detach().cpu().numpy().astype(int)
                private_set = set(PRIVATE_IDX_IN_EXPERIMENT)
                keep_p = np.array([int(t) in private_set for t in y_np], dtype=bool)

                if keep_p.sum() > 0:
                    keep_p_t = torch.as_tensor(keep_p, device=labels_exp.device, dtype=torch.bool)
                    logits_p = logits[keep_p_t]
                    y_p = labels_exp[keep_p_t]
                    preds_p = preds_full[keep_p_t]

                    priv_loss_sum += float(F.cross_entropy(logits_p, y_p, reduction="sum").item())
                    priv_count += int(y_p.numel())
                    priv_correct += int((preds_p == y_p).sum().item())

                    global_set_idx = set(GLOBAL_IDX_IN_EXPERIMENT)
                    preds_p_np = preds_p.detach().cpu().numpy().astype(int)
                    priv_to_global_hits += int(np.sum([int(p) in global_set_idx for p in preds_p_np]))

    if total_full == 0:
        out = {
            "exp_loss": np.nan, "exp_acc": np.nan,
            "global_loss": np.nan, "global_acc": np.nan,
            "global_precision_macro": np.nan, "global_f1_macro": np.nan,
            "unknown_rate_global": np.nan,
        }
        if EXPERIMENT_MODE == "union":
            out.update({"private_loss": np.nan, "private_acc": np.nan, "private_to_global_rate": np.nan})
        return out

    exp_loss = float(total_loss_full / total_full)
    exp_acc = float(correct_full / total_full)

    if global_count == 0:
        global_loss = np.nan
        global_acc = np.nan
        global_precision = np.nan
        global_f1 = np.nan
    else:
        global_loss = float(global_loss_sum / global_count)
        global_acc = float((np.asarray(y_true_g_all) == np.asarray(y_pred_g_all)).mean())
        global_precision = float(precision_score(y_true_g_all, y_pred_g_all, average="macro", zero_division=0))
        global_f1 = float(f1_score(y_true_g_all, y_pred_g_all, average="macro", zero_division=0))

    unknown_rate = _unknown_rate_on_global_examples(np.asarray(y_true_exp), np.asarray(y_pred_exp), local_to_global)

    out = {
        "exp_loss": exp_loss,
        "exp_acc": exp_acc,
        "global_loss": global_loss,
        "global_acc": global_acc,
        "global_precision_macro": global_precision,
        "global_f1_macro": global_f1,
        "unknown_rate_global": unknown_rate,
    }

    # union private-only final
    if EXPERIMENT_MODE == "union":
        if priv_count == 0:
            out.update({"private_loss": np.nan, "private_acc": np.nan, "private_to_global_rate": np.nan})
        else:
            out.update({
                "private_loss": float(priv_loss_sum / priv_count),
                "private_acc": float(priv_correct / priv_count),
                "private_to_global_rate": float(priv_to_global_hits / priv_count),
            })

    return out

def test_global_full_and_global(model: nn.Module, loader: DataLoader):
    return eval_global_basic_plus(model, loader)

# ============================================================
# ADDITIONAL METRICS (FINAL EVALUATION)
# Computes per-class metrics, ROC-AUC, AUPRC, and confusion
# matrices for the shared global classes.
# ============================================================
def compute_additional_metrics(net, testloader):
    net.eval()
    cols_global_in_head, local_to_global = _get_global_mapping_from_loader(testloader)

    y_true_exp, y_pred_exp = [], []
    y_true_g, y_pred_g = [], []
    probs_g_all = []

    with torch.no_grad():
        for X, y_exp in testloader:
            X, y_exp = X.to(DEVICE), y_exp.to(DEVICE)
            logits = net(X)

            preds_exp = torch.argmax(logits, dim=1)
            y_true_exp.extend(y_exp.detach().cpu().numpy())
            y_pred_exp.extend(preds_exp.detach().cpu().numpy())

            keep, y_true_g_np = _filter_to_global_only_from_batch(y_exp, local_to_global)
            if np.any(keep) and (cols_global_in_head is not None):
                keep_t = torch.as_tensor(keep, device=y_exp.device, dtype=torch.bool)
                logits_kept = logits[keep_t]

                preds_g_t, probs_g_t, _ = _global_restricted_preds_and_probs_modeaware(logits_kept, cols_global_in_head)

                y_true_g.extend(y_true_g_np.tolist())
                y_pred_g.extend(preds_g_t.detach().cpu().numpy().tolist())
                probs_g_all.extend(probs_g_t.detach().cpu().numpy())

    cm_full = None
    try:
        cm_full = confusion_matrix(
            np.asarray(y_true_exp, dtype=int),
            np.asarray(y_pred_exp, dtype=int),
            labels=list(range(NUM_EXPERIMENT_CLASSES)),
        )
    except Exception:
        cm_full = None

    if len(y_true_g) == 0:
        return {
            "precision": [], "recall": [], "f1": [],
            "auc_roc": [], "auprc": [],
            "cm_global": None,
            "cm_full": cm_full,
            "unknown_rate_global": _unknown_rate_on_global_examples(np.asarray(y_true_exp), np.asarray(y_pred_exp), local_to_global),
        }

    labels_global = list(range(len(GLOBAL_CLASSES)))
    precision = precision_score(y_true_g, y_pred_g, average=None, labels=labels_global, zero_division=0)
    recall = recall_score(y_true_g, y_pred_g, average=None, labels=labels_global, zero_division=0)
    f1v = f1_score(y_true_g, y_pred_g, average=None, labels=labels_global, zero_division=0)

    probs_g_all = np.asarray(probs_g_all, dtype=float)

    auc_roc, auprc = [], []
    for c in labels_global:
        y_true_c = (np.asarray(y_true_g) == c).astype(int)
        y_score_c = probs_g_all[:, c]
        try:
            auc_roc.append(roc_auc_score(y_true_c, y_score_c))
        except Exception:
            auc_roc.append(np.nan)
        try:
            auprc.append(average_precision_score(y_true_c, y_score_c))
        except Exception:
            auprc.append(np.nan)

    cm_global = confusion_matrix(y_true_g, y_pred_g, labels=labels_global)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1v,
        "auc_roc": auc_roc,
        "auprc": auprc,
        "cm_global": cm_global,
        "cm_full": cm_full,
        "unknown_rate_global": _unknown_rate_on_global_examples(np.asarray(y_true_exp), np.asarray(y_pred_exp), local_to_global),
    }

# ============================================================
# CSV LOGGING UTILITIES
# Save per-round metrics for each client and dataset split.
# ============================================================
def pick(*vals):
    for v in vals:
        if v is None:
            continue
        try:
            if np.isnan(v):
                continue
        except Exception:
            pass
        return v
    return np.nan

def save_metrics_to_csv(partition_id, round_idx, results, per_round_global, per_round_private, dataset_type="train"):
    def write_row(path: str, row: dict):
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    # ---------------- GLOBAL CSV ----------------
    if isinstance(results, dict):
        path = os.path.join(per_round_global, f"client_{partition_id}_{dataset_type}.csv")
        row = {"round": int(round_idx)}

        # full exp
        if "exp_loss" in results or "exp_acc" in results:
            row["exp_loss"] = pick(results.get("exp_loss", None))
            row["exp_acc"] = pick(results.get("exp_acc", None))

        # global-only
        if "global_loss" in results or "global_acc" in results:
            row["global_loss"] = pick(results.get("global_loss", None))
            row["global_acc"] = pick(results.get("global_acc", None))

        if "global_precision_macro" in results:
            row["global_precision_macro"] = pick(results.get("global_precision_macro", None))
        if "global_f1_macro" in results:
            row["global_f1_macro"] = pick(results.get("global_f1_macro", None))
        if "unknown_rate_global" in results:
            row["unknown_rate_global"] = pick(results.get("unknown_rate_global", None))

        # union private-only
        if "private_loss" in results:
            row["private_loss"] = pick(results.get("private_loss", None))
        if "private_acc" in results:
            row["private_acc"] = pick(results.get("private_acc", None))
        if "private_to_global_rate" in results:
            row["private_to_global_rate"] = pick(results.get("private_to_global_rate", None))

        # training dict format
        if "global" in results:
            g = results.get("global", {})
            if "exp_train_loss" in g:
                row["exp_loss"] = pick(row.get("exp_loss", None), g.get("exp_train_loss"))
            if "exp_train_acc" in g:
                row["exp_acc"] = pick(row.get("exp_acc", None), g.get("exp_train_acc"))

        if len(row.keys()) > 1:
            write_row(path, row)

    # ---------------- PRIVATE CSV ----------------
    if isinstance(results, dict) and ("private" in results or "private_loss" in results or "private_acc" in results):
        path = os.path.join(per_round_private, f"client_{partition_id}_{dataset_type}.csv")

        loss = pick(results.get("private", {}).get("train_loss"), results.get("private_loss"))
        acc = pick(results.get("private", {}).get("train_acc"), results.get("private_acc"))

        row = {"round": int(round_idx), "loss": loss, "accuracy": acc}
        write_row(path, row)

def sanitize_metrics(d: dict) -> dict:
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.detach().cpu().item()
            else:
                continue
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, bool):
            out[str(k)] = bool(v); continue
        if isinstance(v, int):
            out[str(k)] = int(v); continue
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                continue
            out[str(k)] = float(v); continue
        if isinstance(v, str):
            out[str(k)] = str(v); continue
    return out

# ============================================================
# PRIVATE CLASSIFIER (ISOLATION MODE)
# Local classifier trained on client-specific private classes.
# ============================================================
def private_forward_from_global_backbone(global_model: nn.Module, classifier: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = global_model.backbone.features(x)
    out = F.relu(feats, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    return classifier(out)

def train_private_classifier(global_model: nn.Module, classifier: nn.Module, loader: DataLoader, epochs: int) -> dict:
    global_model.eval()
    classifier.train()

    ds = getattr(loader, "dataset", None)
    targets = None
    try:
        if isinstance(ds, Subset):
            base = ds.dataset
            idxs = np.asarray(ds.indices, dtype=int)
            if hasattr(base, "targets"):
                base_targets = np.asarray(base.targets, dtype=int)
                targets = base_targets[idxs]
        else:
            if hasattr(ds, "targets"):
                targets = np.asarray(ds.targets, dtype=int)
    except Exception:
        targets = None

    if targets is None or targets.size == 0:
        return {"train_loss": np.nan, "train_acc": np.nan}

    num_classes = classifier.out_features
    class_counts = np.bincount(targets.astype(int), minlength=num_classes)

    weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(classifier.parameters())

    total_loss, total_correct, total_samples = 0.0, 0, 0
    for _ in range(epochs):
        for imgs, y in loader:
            imgs, y = imgs.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = private_forward_from_global_backbone(global_model, classifier, imgs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

    return {
        "train_loss": total_loss / max(total_samples, 1),
        "train_acc": total_correct / max(total_samples, 1),
    }

def eval_private_classifier(global_model: nn.Module, classifier: nn.Module, loader: DataLoader) -> dict:
    global_model.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for imgs, y in loader:
            imgs, y = imgs.to(DEVICE), y.to(DEVICE)
            logits = private_forward_from_global_backbone(global_model, classifier, imgs)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

    if total_samples == 0:
        return {"loss": np.nan, "acc": np.nan}
    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}

# ============================================================
# FEDERATED CLIENT
# Implements the Flower NumPyClient interface.
# Handles:
# - local training
# - validation
# - metric reporting
# ============================================================
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.global_model = GlobalModel(NUM_EXPERIMENT_CLASSES, load_pretrained=True).to(DEVICE)

        self.private_classifier = None
        self.private_num_classes = None

    def _get_loaders(self):
        data = CLIENT_DATASETS[self.partition_id]
        return (
            data["global_train"], data["global_val"], data["global_test_final"],
            data["private_train"], data["private_val"], data["private_test_final"]
        )

    def _ensure_private_classifier(self, private_loader: DataLoader):
        if private_loader is None:
            return
        dataset_base = private_loader.dataset
        if hasattr(dataset_base, "dataset"):
            dataset_base = dataset_base.dataset
        num_private_classes = len(getattr(dataset_base, "class_to_idx", {}))
        if num_private_classes <= 0:
            return
        if (self.private_classifier is None) or (self.private_num_classes != num_private_classes):
            self.private_classifier = nn.Linear(
                self.global_model.backbone.classifier.in_features,
                num_private_classes
            ).to(DEVICE)
            self.private_num_classes = num_private_classes

            ckpt = private_ckpt_path(self.partition_id)
            if os.path.exists(ckpt):
                try:
                    self.private_classifier.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                except Exception:
                    pass

    def get_parameters(self, config):
        global_train, _, _, _, _, _ = self._get_loaders()
        if global_train is None:
            return []
        return get_parameters(self.global_model)

    def fit(self, parameters, config):
        round_idx = int(config.get("round", 0))
        global_epochs = int(config.get("global_epochs", GLOBAL_EPOCHS))
        private_epochs = int(config.get("private_epochs", PRIVATE_EPOCHS))

        global_train, _, _, private_train, _, _ = self._get_loaders()
        if global_train is None:
            return [], 0, {}

        if parameters is not None and len(parameters) > 0:
            set_parameters(self.global_model, parameters)

        global_params = [p.detach().clone() for p in self.global_model.parameters()]
        proximal_mu = float(config.get("proximal_mu", PROXIMAL_MU))

        results_train = train_global(
            global_model=self.global_model,
            global_loader=global_train,
            global_params=global_params,
            proximal_mu=proximal_mu,
            epochs=global_epochs
        )

        # Private classifier used only in isolation mode
        if private_train is not None:
            self._ensure_private_classifier(private_train)
            if self.private_classifier is not None and (self.private_num_classes or 0) > 0:
                priv_train = train_private_classifier(
                    self.global_model, self.private_classifier, private_train, epochs=private_epochs
                )
                results_train["private"] = {"train_loss": priv_train["train_loss"], "train_acc": priv_train["train_acc"]}
                try:
                    torch.save(self.private_classifier.state_dict(), private_ckpt_path(self.partition_id))
                except Exception:
                    pass

        save_metrics_to_csv(self.partition_id, round_idx, results_train, per_round_global, per_round_private, "train")

        metrics_raw = {}
        if "global" in results_train:
            g = results_train["global"]
            metrics_raw = {
                "exp_loss": float(g.get("exp_train_loss", np.nan)),
                "exp_acc": float(g.get("exp_train_acc", np.nan)),
            }

        metrics_out = sanitize_metrics(metrics_raw)
        num_examples = int(len(global_train.dataset))
        return get_parameters(self.global_model), num_examples, metrics_out

    def evaluate(self, parameters, config):
        round_idx = int(config.get("round", 0))
        _, global_val, global_test_final, _, private_val, private_test_final = self._get_loaders()

        if global_val is None:
            return float("nan"), 0, {}

        if parameters is not None and len(parameters) > 0:
            set_parameters(self.global_model, parameters)

        # --- GLOBAL VALIDATION ---
        results_val = eval_global_basic_plus(self.global_model, global_val)
        save_metrics_to_csv(self.partition_id, round_idx, results_val, per_round_global, per_round_private, "val")

        # --- PRIVATE VALIDATION (only if available) ---
        if private_val is not None:
            self._ensure_private_classifier(private_val)
            if self.private_classifier is not None:
                pv = eval_private_classifier(self.global_model, self.private_classifier, private_val)
                priv_val_results = {"private_loss": float(pv["loss"]), "private_acc": float(pv["acc"])}
                save_metrics_to_csv(self.partition_id, round_idx, priv_val_results, per_round_global, per_round_private, "private_val")

        metrics_raw = {
            "exp_loss": float(results_val.get("exp_loss", np.nan)),
            "exp_acc": float(results_val.get("exp_acc", np.nan)),
            "global_loss": float(results_val.get("global_loss", np.nan)),
            "global_acc": float(results_val.get("global_acc", np.nan)),
            "global_precision_macro": float(results_val.get("global_precision_macro", np.nan)),
            "global_f1_macro": float(results_val.get("global_f1_macro", np.nan)),
            "unknown_rate_global": float(results_val.get("unknown_rate_global", np.nan)),
        }

        # Union-mode private-only metrics reported to the server (val)
        if EXPERIMENT_MODE == "union":
            metrics_raw["val_private_loss"] = float(results_val.get("private_loss", np.nan))
            metrics_raw["val_private_acc"] = float(results_val.get("private_acc", np.nan))
            metrics_raw["val_private_to_global_rate"] = float(results_val.get("private_to_global_rate", np.nan))

        # --- GLOBAL TEST_FINAL (last round only) ---
        if (global_test_final is not None) and (round_idx == NUM_ROUNDS):
            res_tf = test_global_full_and_global(self.global_model, global_test_final)
            save_metrics_to_csv(self.partition_id, round_idx, res_tf, per_round_global, per_round_private, "test_final")

            metrics_raw["test_final_exp_loss"] = float(res_tf.get("exp_loss", np.nan))
            metrics_raw["test_final_exp_acc"] = float(res_tf.get("exp_acc", np.nan))
            metrics_raw["test_final_global_loss"] = float(res_tf.get("global_loss", np.nan))
            metrics_raw["test_final_global_acc"] = float(res_tf.get("global_acc", np.nan))
            metrics_raw["test_final_global_precision_macro"] = float(res_tf.get("global_precision_macro", np.nan))
            metrics_raw["test_final_global_f1_macro"] = float(res_tf.get("global_f1_macro", np.nan))
            metrics_raw["test_final_unknown_rate_global"] = float(res_tf.get("unknown_rate_global", np.nan))

            # Union-mode private-only metrics in global test_final
            if EXPERIMENT_MODE == "union":
                metrics_raw["test_final_private_loss"] = float(res_tf.get("private_loss", np.nan))
                metrics_raw["test_final_private_acc"] = float(res_tf.get("private_acc", np.nan))
                metrics_raw["test_final_private_to_global_rate"] = float(res_tf.get("private_to_global_rate", np.nan))

        # Isolation mode: private head test_final evaluation (last round only)
        if (private_test_final is not None) and (round_idx == NUM_ROUNDS):
            self._ensure_private_classifier(private_test_final)
            if self.private_classifier is not None:
                ptf = eval_private_classifier(self.global_model, self.private_classifier, private_test_final)
                priv_tf_results = {"private_loss": float(ptf["loss"]), "private_acc": float(ptf["acc"])}
                save_metrics_to_csv(self.partition_id, round_idx, priv_tf_results, per_round_global, per_round_private, "private_test_final")
                metrics_raw["private_test_final_loss"] = float(ptf["loss"])
                metrics_raw["private_test_final_acc"] = float(ptf["acc"])

        metrics_out = sanitize_metrics(metrics_raw)
        num_examples = int(len(global_val.dataset))

        # Loss reported to the server: global_loss
        loss_for_server = metrics_out.get("global_loss", float("nan"))
        return float(loss_for_server) if loss_for_server is not None else float("nan"), num_examples, metrics_out


# ============================================================
# DATASET INITIALIZATION
# Load and preprocess datasets for all clients before starting
# the FL simulation.
# ============================================================
for partition_id, client_dir in enumerate(CLIENT_DATA_DIRS):
    load_datasets(partition_id, client_dir, BATCH_SIZE)

# ============================================================
# SERVER-SIDE UTILITIES
# Functions for aggregating metrics across clients and
# logging server-level statistics.
# ============================================================
def _to_int(x):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, np.generic):
        try:
            return int(x.item())
        except Exception:
            return None
    try:
        return int(x)
    except Exception:
        return None

def _metrics_to_dict(m):
    if m is None:
        return None
    if isinstance(m, Mapping):
        return dict(m)
    if isinstance(m, (list, tuple)):
        try:
            return dict(m)
        except Exception:
            return None
    return None

def weighted_average_from_results(results, keys):
    out = {}
    for k in keys:
        num, den = 0.0, 0.0
        for item in results or []:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                continue
            _, res = item

            n = _to_int(getattr(res, "num_examples", None))
            if n is None or n <= 0:
                continue

            m = _metrics_to_dict(getattr(res, "metrics", None))
            if not m:
                continue

            v = m.get(k, None)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            if np.isnan(v) or np.isinf(v):
                continue

            num += n * v
            den += n

        out[k] = (num / den) if den > 0 else float("nan")
    return out

def _write_csv_row(path: str, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# ============================================================ 
# CUSTOM FEDPROX STRATEGY 
# Extends Flower's FedProx strategy to store aggregated 
# parameters and compute server-side metrics. 
# ============================================================
class FedProxWithParams(FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return aggregated

        params_agg, metrics_agg = aggregated
        self.final_parameters = params_agg

        forced = weighted_average_from_results(results, ["exp_loss", "exp_acc"])
        row = {
            "round": int(server_round),
            "exp_train_loss": float(forced.get("exp_loss", np.nan)),
            "exp_train_acc": float(forced.get("exp_acc", np.nan)),
            "n_clients_used": int(len(results)) if results is not None else 0,
            "n_failures": int(len(failures)) if failures is not None else 0,
        }
        _write_csv_row(server_fit_metrics_path, row)

        if not isinstance(metrics_agg, dict):
            metrics_agg = {}
        for k, v in forced.items():
            try:
                v = float(v)
            except Exception:
                continue
            if np.isnan(v) or np.isinf(v):
                continue
            metrics_agg[k] = v

        return params_agg, metrics_agg

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            return aggregated

        loss_agg, metrics_agg = aggregated
        if not isinstance(metrics_agg, dict):
            metrics_agg = {}

        # Note: "private_*" metrics here only apply in aligned union mode.
        # They correspond to private-only metrics from the global head (not the isolation private head).
        forced = weighted_average_from_results(
            results,
            [
                "global_loss", "global_acc", "global_precision_macro", "global_f1_macro", "unknown_rate_global",
                "exp_loss", "exp_acc",
                # union private-only:
                "private_loss", "private_acc", "private_to_global_rate",
            ]
        )

        for k, v in forced.items():
            try:
                v = float(v)
            except Exception:
                continue
            if np.isnan(v) or np.isinf(v):
                continue
            metrics_agg[k] = v

        row = {
            "round": int(server_round),
            "loss_agg": float(loss_agg) if loss_agg is not None else np.nan,

            "global_loss": float(metrics_agg.get("global_loss", np.nan)),
            "global_acc": float(metrics_agg.get("global_acc", np.nan)),
            "global_precision_macro": float(metrics_agg.get("global_precision_macro", np.nan)),
            "global_f1_macro": float(metrics_agg.get("global_f1_macro", np.nan)),
            "unknown_rate_global": float(metrics_agg.get("unknown_rate_global", np.nan)),

            "exp_loss": float(metrics_agg.get("exp_loss", np.nan)),
            "exp_acc": float(metrics_agg.get("exp_acc", np.nan)),

            # union private-only
            "private_loss": float(metrics_agg.get("private_loss", np.nan)),
            "private_acc": float(metrics_agg.get("private_acc", np.nan)),
            "private_to_global_rate": float(metrics_agg.get("private_to_global_rate", np.nan)),

            "n_results": int(len(results)) if results is not None else 0,
            "n_failures": int(len(failures)) if failures is not None else 0,
        }
        _write_csv_row(server_metrics_path, row)

        return loss_agg, metrics_agg

def client_fn(cid: str) -> Client:
    return FlowerClient(int(cid)).to_client()

def fit_config(server_round: int):
    return {
        "round": int(server_round),
        "global_epochs": int(GLOBAL_EPOCHS),
        "private_epochs": int(PRIVATE_EPOCHS),
        "proximal_mu": float(PROXIMAL_MU),
    }

def eval_config(server_round: int):
    return {"round": int(server_round)}

# ============================================================ 
# START FEDERATED SIMULATION 
# Launch Flower simulation with the configured strategy 
# and client resources. 
# ============================================================

strategy = FedProxWithParams(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    fraction_evaluate=1.0,
    min_evaluate_clients=NUM_CLIENTS,
    proximal_mu=float(PROXIMAL_MU),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
)

hist = start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=flwr.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 1},
)

# ============================================================
# FINAL EVALUATION 
# After training completes, evaluate the final global model 
# on each client test set and export detailed metrics. 
# ============================================================
from flwr.common import parameters_to_ndarrays

def _infer_private_num_classes_from_loader(private_loader: DataLoader) -> int:
    if private_loader is None:
        return 0
    ds = private_loader.dataset
    if hasattr(ds, "dataset"):
        ds = ds.dataset
    return int(len(getattr(ds, "class_to_idx", {}) or {}))

def save_final_metrics():
    if not hasattr(strategy, "final_parameters") or strategy.final_parameters is None:
        print("[WARN] Final global parameters were not found.")
        return

    final_global_params = parameters_to_ndarrays(strategy.final_parameters)

    def safe_array(x, n):
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size < n:
            arr = np.concatenate([arr, np.full(n - arr.size, np.nan)])
        elif arr.size > n:
            arr = arr[:n]
        return arr

    for partition_id in range(NUM_CLIENTS):
        client_data = CLIENT_DATASETS.get(partition_id, {})
        global_test_final = client_data.get("global_test_final")

        if global_test_final is None:
            print(f"[INFO] Client {partition_id} has no global_test_final dataset.")
            continue

        # Model initialized with final global parameters
        global_model = GlobalModel(NUM_EXPERIMENT_CLASSES, load_pretrained=False).to(DEVICE)
        set_parameters(global_model, final_global_params)

        # --- global test_final (includes union private-only metrics if applicable) ---
        res_tf = test_global_full_and_global(global_model, global_test_final)

        tf_path = os.path.join(final_eval_global, f"client_{partition_id}_test_final_metrics.csv")
        pd.DataFrame([{
            "client": partition_id,
            "mode": EXPERIMENT_MODE,
            **res_tf
        }]).to_csv(tf_path, index=False)

        # --- GLOBAL PER-CLASS METRICS + CONFUSION MATRICES ---
        metrics = compute_additional_metrics(global_model, global_test_final)

        precision = safe_array(metrics.get("precision", []), len(GLOBAL_CLASSES))
        recall = safe_array(metrics.get("recall", []), len(GLOBAL_CLASSES))
        f1v = safe_array(metrics.get("f1", []), len(GLOBAL_CLASSES))
        auc_roc = safe_array(metrics.get("auc_roc", []), len(GLOBAL_CLASSES))
        auprc = safe_array(metrics.get("auprc", []), len(GLOBAL_CLASSES))

        df = pd.DataFrame({
            "class_idx": list(range(len(GLOBAL_CLASSES))),
            "class_name": GLOBAL_CLASSES,
            "precision": precision,
            "recall": recall,
            "f1": f1v,
            "auc_roc": auc_roc,
            "auprc": auprc,
        })
        df.to_csv(os.path.join(final_eval_global, f"client_{partition_id}_final_metrics.csv"), index=False)

        cm_g = metrics.get("cm_global", None)
        if cm_g is not None:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_g, annot=True, fmt="d", cmap="Blues",
                        xticklabels=GLOBAL_CLASSES, yticklabels=GLOBAL_CLASSES)
            plt.title(f"GLOBAL-only Confusion Matrix (restricted) Client {partition_id} [{EXPERIMENT_MODE}]")
            plt.xlabel("Predicted (global)")
            plt.ylabel("True (global)")
            plt.tight_layout()
            plt.savefig(os.path.join(final_eval_global, f"client_{partition_id}_confusion_matrix_GLOBAL.png"))
            plt.close()

        cm_full = metrics.get("cm_full", None)
        if cm_full is not None:
            labels_full = [IDX_TO_EXPERIMENT_CLASS[i] for i in range(NUM_EXPERIMENT_CLASSES)]
            plt.figure(figsize=(max(6, 0.55 * NUM_EXPERIMENT_CLASSES), max(5, 0.55 * NUM_EXPERIMENT_CLASSES)))
            sns.heatmap(cm_full, annot=False, fmt="d", cmap="Blues",
                        xticklabels=labels_full, yticklabels=labels_full)
            plt.title(f"FULL Confusion Matrix (experiment) Client {partition_id} [{EXPERIMENT_MODE}]")
            plt.xlabel("Predicted (experiment)")
            plt.ylabel("True (experiment)")
            plt.tight_layout()
            plt.savefig(os.path.join(final_eval_global, f"client_{partition_id}_confusion_matrix_FULL.png"))
            plt.close()

        # --- ISOLATION MODE: PRIVATE TEST_FINAL (private head, if available) ---
        if EXPERIMENT_MODE == "isolation":
            private_test_final = client_data.get("private_test_final", None)
            n_priv = _infer_private_num_classes_from_loader(private_test_final)
            ckpt = private_ckpt_path(partition_id)

            if (private_test_final is not None) and (n_priv > 0) and os.path.exists(ckpt):
                try:
                    private_classifier = nn.Linear(
                        global_model.backbone.classifier.in_features,
                        n_priv
                    ).to(DEVICE)
                    private_classifier.load_state_dict(torch.load(ckpt, map_location=DEVICE))

                    res_priv_tf = eval_private_classifier(global_model, private_classifier, private_test_final)

                    outp = {
                        "client": partition_id,
                        "mode": EXPERIMENT_MODE,
                        "private_test_final_loss": float(res_priv_tf.get("loss", np.nan)),
                        "private_test_final_acc": float(res_priv_tf.get("acc", np.nan)),
                        "n_private_classes": int(n_priv),
                    }
                    priv_path = os.path.join(final_eval_private, f"client_{partition_id}_private_test_final_metrics.csv")
                    pd.DataFrame([outp]).to_csv(priv_path, index=False)
                except Exception:
                    pass

        # diagnostics
        try:
            with open(os.path.join(final_eval_global, f"client_{partition_id}_diagnostics.txt"), "w") as f:
                f.write(f"unknown_rate_global={metrics.get('unknown_rate_global', np.nan)}\n")
                f.write(f"NUM_EXPERIMENT_CLASSES={NUM_EXPERIMENT_CLASSES}\n")
                f.write(f"EXPERIMENT_CLASSES={EXPERIMENT_CLASSES}\n")
                f.write("\n--- test_final (paper metrics) ---\n")
                for k in ["global_acc", "global_precision_macro", "global_f1_macro", "global_loss",
                          "exp_acc", "exp_loss", "unknown_rate_global",
                          # union private-only
                          "private_acc", "private_loss", "private_to_global_rate"]:
                    if k in res_tf:
                        f.write(f"{k}={res_tf.get(k, np.nan)}\n")
        except Exception:
            pass

    print("[INFO] Final evaluation metrics saved (global per-class metrics, confusion matrices, per-client test_final, and isolation private test_final).")

def build_server_summary_from_test_final(save_path: str):
    rows = []

    for cid in range(NUM_CLIENTS):
        tf_path = os.path.join(final_eval_global, f"client_{cid}_test_final_metrics.csv")
        if not os.path.exists(tf_path):
            rows.append({"client": cid, "has_test_final": 0})
            continue

        try:
            df = pd.read_csv(tf_path)
        except Exception:
            rows.append({"client": cid, "has_test_final": 0})
            continue

        if df is None or df.empty:
            rows.append({"client": cid, "has_test_final": 0})
            continue

        last = df.iloc[-1].to_dict()
        row = {"client": cid, "has_test_final": 1}

        for k in [
            "exp_loss", "exp_acc",
            "global_loss", "global_acc",
            "global_precision_macro", "global_f1_macro",
            "unknown_rate_global",
            # union private-only (if applicable):
            "private_loss", "private_acc", "private_to_global_rate",
        ]:
            if k in last:
                row[k] = last[k]

        rows.append(row)

    dfc = pd.DataFrame(rows)

    df_ok = dfc[dfc["has_test_final"] == 1].copy()

    metric_cols = [c for c in [
        "global_acc", "global_precision_macro", "global_f1_macro", "global_loss",
        "exp_acc", "exp_loss", "unknown_rate_global",
        "private_acc", "private_loss", "private_to_global_rate",
    ] if c in df_ok.columns]

    summary = {"n_clients_with_test_final": int(df_ok.shape[0])}

    for c in metric_cols:
        vals = pd.to_numeric(df_ok[c], errors="coerce")
        summary[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
        summary[f"{c}_std"]  = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        summary[f"{c}_min"]  = float(vals.min()) if vals.notna().any() else np.nan
        summary[f"{c}_max"]  = float(vals.max()) if vals.notna().any() else np.nan

    dfs = pd.DataFrame([summary])

    dfc.to_csv(save_path.replace(".csv", "_per_client.csv"), index=False)
    dfs.to_csv(save_path, index=False)

    print(f"[INFO] Server summary saved to: {save_path}")
    print(f"[INFO] Per-client test_final results saved to: {save_path.replace('.csv','_per_client.csv')}")

# ============================================================ 
# PAPER TABLE EXPORT 
# Generate summary tables and LaTeX-ready results for 
# inclusion in the research paper. 
# ============================================================
def export_paper_table(save_csv_path: str, save_latex_path: str = None):
    rows = []

    for cid in range(NUM_CLIENTS):
        tf_path = os.path.join(final_eval_global, f"client_{cid}_test_final_metrics.csv")
        if not os.path.exists(tf_path):
            continue

        try:
            dfc = pd.read_csv(tf_path)
        except Exception:
            continue

        if dfc is None or dfc.empty:
            continue

        last = dfc.iloc[-1].to_dict()

        row = {
            "client": cid,
            "mode": last.get("mode", EXPERIMENT_MODE),

            "global_acc": last.get("global_acc", np.nan),
            "global_precision_macro": last.get("global_precision_macro", np.nan),
            "global_f1_macro": last.get("global_f1_macro", np.nan),
            "global_loss": last.get("global_loss", np.nan),

            "exp_acc": last.get("exp_acc", np.nan),
            "exp_loss": last.get("exp_loss", np.nan),

            "unknown_rate_global": last.get("unknown_rate_global", np.nan),
        }

        # union private-only (if applicable)
        if "private_acc" in last:
            row["private_acc"] = last.get("private_acc", np.nan)
            row["private_loss"] = last.get("private_loss", np.nan)
            row["private_to_global_rate"] = last.get("private_to_global_rate", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No rows available for the paper table (client_*_test_final_metrics.csv not found in final_evaluation).")
        return

    for c in df.columns:
        if c not in ["client", "mode"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(4)

    df.to_csv(save_csv_path, index=False)
    print(f"[INFO] Paper table CSV saved to: {save_csv_path}")

    if save_latex_path is not None:
        latex = df.to_latex(index=False)
        with open(save_latex_path, "w") as f:
            f.write(latex)
        print(f"[INFO] Paper table LaTeX saved to: {save_latex_path}")

# ============================================================ 
# CLEAN SHUTDOWN 
# Safely terminate Ray processes after simulation. 
# ============================================================

def safe_ray_shutdown():
    try:
        import ray
        ray.shutdown()
    except Exception:
        pass

# 1) Final per-client metrics
save_final_metrics()

# 2) Summary server-level
server_summary_path = os.path.join(global_dir, "server_summary_test_final.csv")
build_server_summary_from_test_final(server_summary_path)

# 3) Paper table
paper_csv = os.path.join(global_dir, "paper_table_test_final.csv")
paper_tex = os.path.join(global_dir, "paper_table_test_final.tex")
export_paper_table(paper_csv, save_latex_path=paper_tex)

# 4) Shutdown
safe_ray_shutdown()
