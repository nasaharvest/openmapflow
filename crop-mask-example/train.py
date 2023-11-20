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

from openmapflow.bands import BANDS_MAX  # to add STATISTICS_BANDS_MAX, STATISTICS_BANDS_MIN to openmapflow.bands (can't do it in colab for some reason)
from openmapflow.constants import SUBSET
from openmapflow.pytorch_dataset import PyTorchDataset
from openmapflow.train_utils import (
    generate_model_name,
    get_x_y,
    model_path_from_name,
    upsample_df,
    # upsample_balance_label_country_df
)
from openmapflow.utils import tqdm
from openmapflow.bands import BANDS

try:
    import google.colab  # noqa
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

warnings.simplefilter("ignore", UserWarning)  # TorchScript throws excessive warnings

import logging
import os
import shutil
import io


STATISTICS_BANDS_MAX = [18.0, 2.0, 13259.0, 12729.0, 13347.0, 13405.0, 13412.0, 13479.0, 13301.0, 13334.0, 10735.0, 12617.0, 13265.0, 307.0, 1, 757.0, 41.0, 1.0]
STATISTICS_BANDS_MIN = [-30.0, -45.0, 801.0, 616.0, 355.0, 451.0, 426.0, 426.0, 353.0, 348.0, 54.0, 80.0, 22.0, 295.0, 0.0, 0.0, 0.0, 0.0]


def main(model_name, start_month, input_months, num_epochs, batch_size, lr, upsample_minority_ratio,
         selected_bands=None, lr_step=5, lr_decay=0.1, pre_trained_model_pt="", pre_trained_on_togo=True,
         use_schedule=False):
    # ------------ Dataloaders -------------------------------------
    df = pd.concat([d.load_df() for d in datasets])
    df[label_col] = (df[label_col] > 0.5).astype(int)
    if pre_trained_model_pt and pre_trained_on_togo:
        train_df = df[(df[SUBSET] == "training") & (df["country"] == "Togo")]
        logging.info(f"\n***Fine-Tuning model for Togo country- for {num_epochs} more epochs...\n")
    else:
        train_df = df[df[SUBSET] == "training"]
    
    train_df = upsample_df(train_df, label_col, upsample_minority_ratio)
    # train_df = upsample_balance_label_country_df(train_df, label_col, upsample_minority_ratio)

    val_df = df[df[SUBSET] == "validation"]
    x_train, y_train = get_x_y(train_df, label_col, start_month, input_months)
    x_val, y_val = get_x_y(val_df, label_col, start_month, input_months)

    # Convert to tensors
    train_data = PyTorchDataset(x=x_train, y=y_train)
    val_data = PyTorchDataset(x=x_val, y=y_val)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # ------------ Model -----------------------------------------
    class Model(torch.nn.Module):
        def __init__(self, norm_max=STATISTICS_BANDS_MAX, norm_min=STATISTICS_BANDS_MIN, selected_bands=None):
            super().__init__()
            logging.info("Normalizing band values by STATISTICS_BANDS_MAX")
            if not selected_bands:
                self.selected_bands = list(range(len(BANDS)))
            else:
                self.selected_bands = selected_bands
            num_bands = len(self.selected_bands)

            self.model = TransformerModel(c_in=num_bands, c_out=1)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.norm_max = torch.tensor(norm_min).to(self.device)
            self.norm_min = torch.tensor(norm_max).to(self.device)

        def forward(self, x):
            with torch.no_grad():
                x = (x - self.norm_min) / (self.norm_max - self.norm_min)
                # x = x.clip(0.0, 1.0)  # torch.tanh(x)  # torch.sigmoid(x)
                x = x[:, :, self.selected_bands]
                x = x.transpose(2, 1)  # b_sz, n_bands, n_t_series
            x = self.model(x).squeeze(dim=1)
            x = torch.sigmoid(x)
            return x

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if pre_trained_model_pt:
        model = load_script_model(pre_trained_model_pt, device=device)
        logging.info(f"Using Pre-Trained model: {pre_trained_model_pt}\n***")
    else:
        model = Model(selected_bands=selected_bands).to(device)
    
    logging.info(model)

    # ------------ Model hyperparameters -------------------------------------
    params_to_update = model.parameters()
    optimizer = torch.optim.Adam(params_to_update, lr=lr)
    criterion = torch.nn.BCELoss()
    if use_schedule:
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
    
    if model_name == "":
        model_name = generate_model_name(val_df=val_df, start_month=start_month)

    lowest_validation_loss = None
    metrics = {}
    train_batches = 1 + len(train_data) // batch_size
    val_batches = 1 + len(val_data) // batch_size
    logging.info(f"EPOCH: TRAIN-LOSS,\t VAL-LOSS\t --- F1,\t ACCURACY,\t PERCISION,\t RECALL,\t ROC-AUC,\t")
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
            if use_schedule:
                lr_scheduler.step()
            train_loss = total_train_loss / len(train_data)
            val_loss = total_val_loss / len(val_data)

            if lowest_validation_loss is None or val_loss < lowest_validation_loss:
                lowest_validation_loss = val_loss
                metrics = {
                    "f1": f1_score(y_true, y_pred),
                    "accuracy": accuracy_score(y_true, y_pred),
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
                logging.info(f"{epoch}: {train_loss},\t {val_loss}\t --- {metrics['f1']},\t {metrics['accuracy']},\t {metrics['precision']},\t {metrics['recall']},\t {metrics['roc_auc']}")

                if model_path.exists():
                    model_path.unlink()
                else:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                sm.save(str(model_path))

    print(f"MODEL_NAME={model_name}")
    logging.info(f"\nmodel_path: {model_path}")
    print(yaml.dump(metrics, allow_unicode=True, default_flow_style=False))


def load_script_model(model_path, device='cpu'):
    # Load ScriptModule from io.BytesIO object
    with open(model_path, 'rb') as f:
        buffer_data = io.BytesIO(f.read())

    buffer_data.seek(0)
    return torch.jit.load(buffer_data, map_location=torch.device(device))


if __name__ == "__main__":
    # ------------ Arguments -------------------------------------
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--start_month", type=str, default="February")
    parser.add_argument("--input_months", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--upsample_minority_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--selected_bands", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pre_trained_model_pt", type=str, default="")
    parser.add_argument("--pre_trained_on_togo", type=bool, default=True)

    args = parser.parse_args().__dict__

    start_month: str = args["start_month"]
    batch_size: int = args["batch_size"]
    upsample_minority_ratio: float = args["upsample_minority_ratio"]
    num_epochs: int = args["epochs"]
    lr: int = args["lr"]
    lr_step: int = args["lr_step"]
    lr_decay: float = args["lr_decay"]
    model_name: str = args["model_name"]
    input_months: int = args["input_months"]
    pre_trained_model_pt: str = args["pre_trained_model_pt"]
    pre_trained_on_togo: bool = args["pre_trained_on_togo"]
    selected_bands: str = args["selected_bands"]

    if not selected_bands:
        selected_bands = None  # no selection = take all bands
    else:
        selected_bands = list(map(int, selected_bands.split(",")))

    res_dir = "results"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    model_name = args["model_name"]
    if not model_name == "":
        usmr = str(upsample_minority_ratio)
        model_name = f"{model_name}-n_ep{num_epochs}-{start_month[:3]}_{input_months}-lr{str(lr)[0]}{str(lr)[2:]}-b_size{batch_size}-upsm_ratio{usmr[0]}{usmr[2:]}"

        model_dir = f"{res_dir}/{model_name}"
        print(model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        log_file_path = f'{model_dir}/{model_name}.log'
        shutil.copy("naama_train.py", f"{model_dir}/naama_train.py")
    else:
        log_file_path = f'{res_dir}/unknown_model_name.log'

    logging.basicConfig(level=logging.INFO, filename= log_file_path, filemode='w', format='%(message)s')
    logging.info("-----------------------------------------------------")
    logging.info(f"TRAINING PARAMETERS:\nmodel_name: {model_name}\nstart_month: {start_month}")
    logging.info(f"input_months: {input_months}\nbatch_size: {batch_size}")
    logging.info(f"upsample_minority_ratio: {upsample_minority_ratio}\nlr: {lr}")
    logging.info(f"num_epochs: {num_epochs}\n")

    main(model_name, start_month, input_months, num_epochs, batch_size, lr, upsample_minority_ratio,
         selected_bands=selected_bands, lr_step=lr_step, lr_decay=lr_decay,
         pre_trained_model_pt=pre_trained_model_pt, pre_trained_on_togo=pre_trained_on_togo)

    

