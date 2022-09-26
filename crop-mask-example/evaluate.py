"""
Example model evaluation script
"""
from argparse import ArgumentParser

import pandas as pd
import torch
import yaml
from datasets import datasets, label_col
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import SUBSET
from openmapflow.pytorch_dataset import PyTorchDataset
from openmapflow.train_utils import get_x_y, model_path_from_name
from openmapflow.utils import tqdm

# ------------ Arguments -------------------------------------
parser = ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--start_month", type=str, default="February")
parser.add_argument("--input_months", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--skip_yaml", dest="skip_yaml", action="store_true")
parser.set_defaults(skip_yaml=False)

args = parser.parse_args().__dict__
start_month: str = args["start_month"]
input_months: int = args["input_months"]
batch_size: int = args["batch_size"]
model_name: str = args["model_name"]
skip_yaml = bool = args["skip_yaml"]
model_path = model_path_from_name(model_name=model_name)

# ------------ Dataloaders -------------------------------------
df = pd.concat([d.load_df() for d in datasets])
df[label_col] = (df[label_col] > 0.5).astype(int)
test_df = df[df[SUBSET] == "testing"]
x_test, y_test = get_x_y(test_df, label_col, start_month, input_months)

# Convert to tensors
test_data = PyTorchDataset(x=x_test, y=y_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model_pt = torch.jit.load(model_path)
model_pt.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------ Testing / Evaluation -------------------------------
test_batches = 1 + len(test_data) // batch_size
y_true, y_score, y_pred = [], [], []
with torch.no_grad():
    for x in tqdm(test_dataloader, total=test_batches, desc="Testing", leave=False):
        inputs, labels = x[0].to(device), x[1].to(device)

        # Get model outputs
        outputs = model_pt(inputs)
        y_true += labels.tolist()
        y_score += outputs.tolist()
        y_pred += (outputs > 0.5).long().tolist()

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_score),
}
metrics = {k: round(float(v), 4) for k, v in metrics.items()}

all_metrics = {}
if not skip_yaml and (PROJECT_ROOT / DataPaths.METRICS).exists():
    with (PROJECT_ROOT / DataPaths.METRICS).open() as f:
        all_metrics = yaml.safe_load(f)

all_metrics[model_name] = {
    "test_metrics": metrics,
    "test_size": len(test_df),
    label_col: test_df[label_col].value_counts(normalize=True).to_dict(),
}
print(yaml.dump(all_metrics[model_name], allow_unicode=True, default_flow_style=False))

if not skip_yaml:
    with open((PROJECT_ROOT / DataPaths.METRICS), "w") as f:
        yaml.dump(all_metrics, f)
