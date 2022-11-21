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
from tsai.models.TSSequencerPlus import TSSequencerPlus
from tsai.models.RNNPlus import GRUPlus, RNNPlus, LSTMPlus

from openmapflow.bands import BANDS_MAX
from openmapflow.constants import SUBSET
from openmapflow.pytorch_dataset import PyTorchDataset
from openmapflow.train_utils import (
    generate_model_name,
    get_x_y,
    model_path_from_name,
    upsample_df,
    # upsample_augment_df
)
from openmapflow.utils import tqdm

try:
    import google.colab  # noqa

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

warnings.simplefilter("ignore", UserWarning)  # TorchScript throws excessive warnings

import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from naama_train_utils import upsample_balance_label_country_df, load_script_model #, load_ckp


def main(model_name, start_month, input_months, num_epochs, batch_size, lr, upsample_minority_ratio,
lr_step=5, lr_decay=0.1, pre_trained_model_pt=""):
    # ------------ Dataloaders -------------------------------------
    df = pd.concat([d.load_df() for d in datasets])
    df[label_col] = (df[label_col] > 0.5).astype(int)
    if pre_trained_model_pt == "":
        train_df = df[df[SUBSET] == "training"]
        # train_df = upsample_df(train_df, label_col, upsample_minority_ratio)
    else:
        train_df = df[(df[SUBSET] == "training") & (df["country"] == "Togo")]
        # train_df = upsample_balance_label_country_df(train_df, label_col, upsample_minority_ratio)
        logging.info(f"\n***\nUsing Pre-Trained model: {pre_trained_model_pt}")
        logging.info(f"Re-Training model on Togo for {num_epochs} more epochs...\n***\n")
    
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
            # self.model = TSSequencerPlus(c_in=num_bands, c_out=1, seq_len=num_timesteps)
            # self.model = LSTMPlus(c_in=num_bands, c_out=1)
            self.normalization_vals = torch.tensor(normalization_vals)

        def forward(self, x):
            with torch.no_grad():
                x = x / self.normalization_vals
                x = x.transpose(2, 1)
            x = self.model(x).squeeze(dim=1)
            x = torch.sigmoid(x)
            return x

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if pre_trained_model_pt:
        model = load_script_model(pre_trained_model_pt, device=device)
    else:
        model = Model().to(device)
    
    logging.info(model)

    # ------------ Model hyperparameters -------------------------------------
    params_to_update = model.parameters()
    optimizer = torch.optim.Adam(params_to_update, lr=lr)
    criterion = torch.nn.BCELoss()
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
    # if pre_trained_model_pt:
    #     model, optimizer, start_epoch = load_ckp(pre_trained_model_pt, model, optimizer)

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
            # lr_scheduler.step()  #  NAAMA
            train_loss = total_train_loss / len(train_data)
            val_loss = total_val_loss / len(val_data)
            # writer.add_scalar("Loss/train", train_loss, epoch)
            # writer.add_scalar("Loss/val", val_loss, epoch)

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

if __name__ == "__main__":
  # ------------ Arguments -------------------------------------
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--start_month", type=str, default="February")
    parser.add_argument("--input_months", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--upsample_minority_ratio", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_step", type=int, default=5)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pre_trained_model_pt", type=str, default="")

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
    pre_trained_model_pt:str = args["pre_trained_model_pt"]

    model_name = args["model_name"]
    if not model_name == "":
        usmr = str(upsample_minority_ratio)
        model_name = f"{model_name}-n_ep{num_epochs}-{start_month[:3]}_{input_months}-lr{str(lr)[2:]}-b_size{batch_size}-upsm_ratio{usmr[0]}{usmr[2:]}"
        model_dir = f"results/{model_name}"
        print(model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        log_file_path = f'{model_dir}/{model_name}.log'
        shutil.copy("naama_train.py", f"{model_dir}/naama_train.py")
    else:
        log_file_path = f'results/unknown_model_name.log'

    logging.basicConfig(level=logging.INFO, filename= log_file_path, filemode='w', format='%(message)s')
    logging.info("-----------------------------------------------------")
    logging.info(f"TRAINING PARAMETERS:\nmodel_name: {model_name}\nstart_month: {start_month}")
    logging.info(f"input_months: {input_months}\nbatch_size: {batch_size}")
    logging.info(f"upsample_minority_ratio: {upsample_minority_ratio}\nlr: {lr}")
    logging.info(f"num_epochs: {num_epochs}\n")
    
    
    # # Tensorboard writer
    # experiment_id = datetime.now().strftime('%Y-%m-%d_%H-%M')
    # log_dir = f"results/tf_logs/{model_name}_{experiment_id}"
    # writer = SummaryWriter(log_dir=log_dir)

    main(model_name, start_month, input_months, num_epochs, batch_size, lr, upsample_minority_ratio, lr_step, lr_decay, pre_trained_model_pt)
    # print("tensorboard log dir: " + log_dir)
    # writer.flush()
    # writer.close()

    
