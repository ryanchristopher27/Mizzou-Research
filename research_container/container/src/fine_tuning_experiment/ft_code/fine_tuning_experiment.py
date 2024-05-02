# Imports
from torch import nn
import torch
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as evaluate
import json
import time
import csv

from utils.utils import *
from utils.data import *
from utils.models import *

# GLOBAL VARIABLES
TORCH_NUM_JOBS = int(os.environ.get("TORCH_NUM_JOBS", "4"))
TORCH_NUM_EPOCHS = int(os.environ.get("TORCH_NUM_EPOCHS", "2"))
TORCH_NUM_FOLDS = int(os.environ.get("TORCH_NUM_FOLDS", "5"))
FOLD_NUM = int(os.environ.get("FOLD_NUM", "1"))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.8"))
TORCH_MODEL_NAME = os.environ.get("TORCH_MODEL_NAME", "vit_b_16")
TORCH_DATA_NAME = os.environ.get("TORCH_DATA_NAME", "cifar10")
WRITE_RESULTS = bool(os.environ.get("WRITE_RESULTS", False))
OPTIMIZER = os.environ.get("OPTIMIZER", "AdamW")
LOSS_FUNCTION = os.environ.get("LOSS_FUNCTION", "CrossEntropy")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
EXPERIMENT = int(os.environ.get("EXPERIMENT", "1"))


# MAIN FUNCTION
def main():
    start_time = time.time()

    # PRINT JOB INFO
    print(f"Model: {TORCH_MODEL_NAME}")
    print(f"Dataset: {TORCH_DATA_NAME}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Loss Function: {LOSS_FUNCTION}")
    print(f"Learning Rate: {LEARNING_RATE}\n")

    device, on_gpu = cuda_setup()

    train_batch_size = BATCH_SIZE
    test_batch_size = BATCH_SIZE

    num_classes = 10

    if TORCH_DATA_NAME == "cifar10":
        kfold = False
    else:
        kfold = True
        print(f"Fold: {FOLD_NUM}\n")

    train_data_loader, test_data_loader, input_features, num_classes = get_data(
        data_name = TORCH_DATA_NAME,
        model_name = TORCH_MODEL_NAME,
        num_folds = TORCH_NUM_FOLDS,
        fold = FOLD_NUM,
        train_ratio = TRAIN_RATIO,
        train_batch_size = train_batch_size,
        test_batch_size = test_batch_size,
        num_workers = TORCH_NUM_JOBS
    )

    model = get_model(
        model_name=TORCH_MODEL_NAME,
        pretrain_weights=True,
        input_features=input_features,
        num_classes = num_classes
    )

    # create opt and loss
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    if LOSS_FUNCTION == "CrossEntropy":
        loss_function = nn.CrossEntropyLoss()

    model.cuda() if on_gpu else None

    train_loss_per_epoch = np.zeros(TORCH_NUM_EPOCHS)
    train_accuracy_per_epoch = np.zeros(TORCH_NUM_EPOCHS)

    val_loss_per_epoch = np.zeros(TORCH_NUM_EPOCHS)
    val_accuracy_per_epoch = np.zeros(TORCH_NUM_EPOCHS)

    best_epoch = dict(
        epoch = 0,
        train_loss = np.inf,
        val_loss = np.inf,
        train_accuracy = 0,
        val_accuracy = 0,
    )

    for epoch in range(TORCH_NUM_EPOCHS):
        print("===" * 30 + f"\nEpoch [{epoch+1} / {TORCH_NUM_EPOCHS}]")
        epoch_loss = 0
        epoch_correct = 0

        model.train()

        for i, (images, labels) in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc="Training"):
            images = images.cuda() if on_gpu else None
            labels = labels.cuda() if on_gpu else None

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs.float(), labels.long())

            if torch.isnan(loss):
                raise RuntimeError("Loss reached NaN!")

            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs, 1)
            epoch_correct += torch.sum(predictions == labels)
            epoch_loss += loss.item()

        train_ave_loss = epoch_loss / len(train_data_loader)
        train_accuracy = epoch_correct / (len(train_data_loader) * train_batch_size) * 100

        train_loss_per_epoch[epoch] = train_ave_loss
        train_accuracy_per_epoch[epoch] = train_accuracy

        print(f"Train Loss = {train_ave_loss:.4f}, Train Acc = {train_accuracy:.2f}%")


        model.eval()

        val_epoch_loss = 0
        val_epoch_correct = 0

        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Validating"):
                images = images.cuda() if on_gpu else None
                labels = labels.cuda() if on_gpu else None
                outputs = model(images)

                loss = loss_function(outputs.float(), labels.long())

                _, predictions = torch.max(outputs, 1)
                val_epoch_correct += torch.sum(predictions == labels)
                val_epoch_loss += loss.item()

            val_ave_loss = val_epoch_loss / len(test_data_loader)
            val_accuracy = val_epoch_correct / (len(test_data_loader) * test_batch_size) * 100

            if val_accuracy > float(best_epoch["val_accuracy"]):
                best_epoch = dict(
                    epoch = epoch + 1,
                    train_loss = f"{train_ave_loss:.4f}",
                    val_loss = f"{val_ave_loss:.4f}",
                    train_accuracy = train_accuracy.item(),
                    val_accuracy = val_accuracy.item(),
                )

            val_loss_per_epoch[epoch] = val_ave_loss
            val_accuracy_per_epoch[epoch] = val_accuracy

            print(f"Val Loss = {val_ave_loss:.4f}, Val Acc = {val_accuracy:.2f}%")

    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        print("---" * 30 + f"\nRunning Eval")
        for i, (images, lb) in tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Testing"):

            images = images.cuda() if on_gpu else None

            model_outputs = model(images)

            _, preds = torch.max(model_outputs, 1)

            labels.extend(lb.numpy().tolist())

            preds = preds.cpu() if on_gpu else None
            predictions.extend(preds.numpy().tolist())

    acc = metrics.accuracy_score(predictions, labels)
    prec, rec, fscore, _ = evaluate(predictions, labels, average="macro", zero_division=0)

    end_time = time.time()
    execution_time = end_time - start_time # Time in Seconds

    print("*" * 20 + f"""\n
    Accuracy  \t{acc*100:.2f}%
    Precision  \t{prec*100:.2f}%
    Recall  \t{rec*100:.2f}%
    F-1 Score \t{fscore*100:.2f}%
    """)


    results = {
        "Results": {
            "Experiment": EXPERIMENT,
            "Optimizer": OPTIMIZER,
            "Learning_Rate": LEARNING_RATE,
            "Batch_Size": BATCH_SIZE,
            # "Fold_Num" : FOLD_NUM,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": fscore,
            "Execution_Time": execution_time,
        }
    }

    if kfold:
        results["Results"]["Fold_Number"] = FOLD_NUM

    file_path = f"/rchristopher/data/src/fine_tuning_experiment/ft_results/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = f"{TORCH_MODEL_NAME}_{TORCH_DATA_NAME}_results.json"
    csv_file_name = f"{TORCH_MODEL_NAME}_{TORCH_DATA_NAME}_results.csv"

    previous_data = read_json_from_file(file_path + file_name)

    previous_data['Experiments'].append(results)

    data = previous_data

    fieldnames = list(results["Results"].keys())
    file_exists = os.path.isfile(file_path + csv_file_name)
    with open(file_path + csv_file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(results["Results"])

    # if not os.path.exists(file_path + '/plots'):
    #     os.makedirs(file_path + '/plots')

    if WRITE_RESULTS:
        write_results_to_file(data, file_path + file_name)

        # write_2data_plot_to_file(
        #     data_1 = train_loss_per_epoch,
        #     data_1_label = "Training Loss",
        #     data_2 = val_loss_per_epoch,
        #     data_2_label = "Validation Loss",
        #     x_label = "Epoch",
        #     y_label = "Loss",
        #     title = "Loss vs Epoch",
        #     filename = file_path + f"plots/loss_{experiment_id}.png",
        # )

        # write_2data_plot_to_file(
        #     data_1 = train_accuracy_per_epoch,
        #     data_1_label = "Training Accuracy",
        #     data_2 = val_accuracy_per_epoch,
        #     data_2_label = "Validation Accuracy",
        #     x_label = "Epoch",
        #     y_label = "Accuracy",
        #     title = "Accuracy vs Epoch",
        #     filename = file_path + f"plots/accuracy_{experiment_id}.png",
        # )

        print("Results Saved to 'ft_results' Folder")

if __name__ == "__main__":
    main()