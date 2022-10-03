"""
Example model training script
"""
import warnings
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
from tsai.models.TransformerModel import TransformerModel

from openmapflow.bands import BANDS_MAX
from openmapflow.constants import SUBSET
from openmapflow.pytorch_dataset import PyTorchDataset
from openmapflow.train_utils import (
    generate_model_name,
    get_x_y,
    model_path_from_name,
    upsample_df,
)
from openmapflow.utils import tqdm

try:
    import google.colab  # noqa

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


warnings.simplefilter("ignore", UserWarning)  # TorchScript throws excessive warnings

# ------------ Arguments -------------------------------------
parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--start_month", type=str, default="February")
parser.add_argument("--input_months", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--upsample_minority_ratio", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)

args = parser.parse_args().__dict__
start_month: str = args["start_month"]
batch_size: int = args["batch_size"]
upsample_minority_ratio: float = args["upsample_minority_ratio"]
num_epochs: int = args["epochs"]
lr: int = args["lr"]
model_name: str = args["model_name"]
input_months: int = args["input_months"]
# ------------ Dataloaders -------------------------------------
df = pd.concat([d.load_df() for d in datasets])
df[label_col] = (df[label_col] > 0.5).astype(int)
train_df = df[df[SUBSET] == "training"]
train_df = upsample_df(train_df, label_col, upsample_minority_ratio)
val_df = df[df[SUBSET] == "validation"]
x_train, y_train = get_x_y(train_df, label_col, start_month, input_months)
x_val, y_val = get_x_y(val_df, label_col, start_month, input_months)

# Convert to tensors
train_data = PyTorchDataset(x=x_train, y=y_train)
val_data = PyTorchDataset(x=x_val, y=y_val)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# ------------ Model -----------------------------------------
num_timesteps, num_bands = train_data[0][0].shape


class Model(torch.nn.Module):
    def __init__(self, normalization_vals=BANDS_MAX):
        super().__init__()
        self.model = TransformerModel(c_in=num_bands, c_out=1)
        self.normalization_vals = torch.tensor(normalization_vals)

    def forward(self, x):
        with torch.no_grad():
            x = x / self.normalization_vals
            x = x.transpose(2, 1)
        x = self.model(x).squeeze(dim=1)
        x = torch.sigmoid(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

# ------------ Model hyperparameters -------------------------------------
params_to_update = model.parameters()
optimizer = torch.optim.Adam(params_to_update, lr=lr)
criterion = torch.nn.BCELoss()

if model_name == "":
    model_name = generate_model_name(val_df=val_df, start_month=start_month)

lowest_validation_loss = None
metrics = {}
train_batches = 1 + len(train_data) // batch_size
val_batches = 1 + len(val_data) // batch_size

with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:
    for epoch in tqdm_epoch:

        # ------------------------ Training ----------------------------------------
        total_train_loss = 0.0
        model.train()
        for x in tqdm(
            train_dataloader,
            total=train_batches,
            desc="Train",
            leave=False,
            disable=IN_COLAB,
        ):
            inputs, labels = x[0].to(device), x[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(inputs)

        # ------------------------ Validation --------------------------------------
        total_val_loss = 0.0
        y_true = []
        y_score = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for x in tqdm(
                val_dataloader,
                total=val_batches,
                desc="Validate",
                leave=False,
                disable=IN_COLAB,
            ):
                inputs, labels = x[0].to(device), x[1].to(device)

                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * len(inputs)

                y_true += labels.tolist()
                y_score += outputs.tolist()
                y_pred += (outputs > 0.5).long().tolist()

        # ------------------------ Metrics + Logging -------------------------------
        train_loss = total_train_loss / len(train_data)
        val_loss = total_val_loss / len(val_data)

        if lowest_validation_loss is None or val_loss < lowest_validation_loss:
            lowest_validation_loss = val_loss
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "roc_auc": roc_auc_score(y_true, y_score),
            }
            metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        tqdm_epoch.set_postfix(loss=val_loss)

        # ------------------------ Model saving --------------------------
        if lowest_validation_loss == val_loss:
            # Some models in tsai need to be modified to be TorchScriptable
            # https://github.com/timeseriesAI/tsai/issues/561
            sm = torch.jit.script(model)
            model_path = model_path_from_name(model_name=model_name)
            if model_path.exists():
                model_path.unlink()
            else:
                model_path.parent.mkdir(parents=True, exist_ok=True)
            sm.save(str(model_path))

print(f"MODEL_NAME={model_name}")
print(yaml.dump(metrics, allow_unicode=True, default_flow_style=False))
