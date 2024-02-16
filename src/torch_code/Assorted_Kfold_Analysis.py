# Imports
from torch import nn
import torch
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as evaluate

from utils.utils import *
from utils.data import *
from utils.models import *

# GLOBAL VARIABLES
TORCH_NUM_JOBS = int(os.environ.get("TORCH_NUM_JOBS", "4"))
TORCH_NUM_EPOCHS = int(os.environ.get("TORCH_NUM_EPOCHS", "5"))
TORCH_NUM_FOLDS = int(os.environ.get("TORCH_NUM_FOLDS", "5"))
FOLD_NUM = int(os.environ.get("FOLD_NUM", "1"))
TORCH_MODEL_NAME = os.environ.get("TORCH_MODEL_NAME", "resnet18")
TORCH_DATA_NAME = os.environ.get("TORCH_DATA_NAME", "ucmerced_landuse")
WRITE_RESULTS = bool(os.environ.get("WRITE_RESULTS", False))


# MAIN FUNCTION
def main():
    # PRINT JOB INFO
    print(f"Model: {TORCH_MODEL_NAME}")
    print(f"Dataset: {TORCH_DATA_NAME}")
    print(f"Fold: {FOLD_NUM}\n")

    cuda_setup()

    train_batch_size = 128
    test_batch_size = 128

    num_classes = 10

    train_data_loader, test_data_loader, input_features, num_classes = get_k_fold_data(
        data_name = TORCH_DATA_NAME,
        model_name = TORCH_MODEL_NAME,
        num_folds = TORCH_NUM_FOLDS,
        fold = FOLD_NUM,
        train_batch_size = train_batch_size,
        test_batch_size = test_batch_size,
        num_jobs = TORCH_NUM_JOBS
    )

    model = get_model(
        model_name=TORCH_MODEL_NAME,
        pretrain_weights=True,
        input_features=input_features,
        num_classes = num_classes
    )

    # create opt and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.CrossEntropyLoss()

    model.cuda()

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
            images = images.cuda()
            labels = labels.cuda()

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
                images = images.cuda()
                labels = labels.cuda()
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

            model_outputs = model(images.cuda())

            _, preds = torch.max(model_outputs, 1)

            labels.extend(lb.numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())

    acc = metrics.accuracy_score(predictions, labels)
    prec, rec, fscore, _ = evaluate(predictions, labels, average="macro", zero_division=0)

    print("*" * 20 + f"""\n
    Accuracy  \t{acc*100:.2f}%
    Precision  \t{prec*100:.2f}%
    Recall  \t{rec*100:.2f}%
    F-1 Score \t{fscore*100:.2f}%
    """)

    best_epoch["train_accuracy"] = f"{float(best_epoch['train_accuracy']):.4f}%"
    best_epoch["val_accuracy"] = f"{float(best_epoch['val_accuracy']):.4f}%"

    # Write Results
    results = {
        "Fold_Number": FOLD_NUM,
        "Accuracy": f"{acc * 100:.4f}%",
        "Precision": f"{prec * 100:.4f}%",
        "Recall": f"{rec * 100:.4f}%",
        "F-1_Score": f"{fscore * 100:.4f}%",
        "Best_Epoch": best_epoch,
    }

    file_path = f"results/{TORCH_MODEL_NAME}/{TORCH_DATA_NAME}/epochs_{TORCH_NUM_EPOCHS}/fold_{FOLD_NUM}/"
    file_name = f"results_{FOLD_NUM}.json"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if WRITE_RESULTS:
        write_results_to_file(results, file_path + file_name)

        write_2data_plot_to_file(
            data_1 = train_loss_per_epoch,
            data_1_label = "Training Loss",
            data_2 = val_loss_per_epoch,
            data_2_label = "Validation Loss",
            x_label = "Epoch",
            y_label = "Loss",
            title = "Loss vs Epoch",
            filename = file_path + f"loss_{FOLD_NUM}.png",
        )

        write_2data_plot_to_file(
            data_1 = train_accuracy_per_epoch,
            data_1_label = "Training Accuracy",
            data_2 = val_accuracy_per_epoch,
            data_2_label = "Validation Accuracy",
            x_label = "Epoch",
            y_label = "Accuracy",
            title = "Accuracy vs Epoch",
            filename = file_path + f"accuracy_{FOLD_NUM}.png",
        )

        print("Results Saved to 'results' Folder")

if __name__ == "__main__":
    main()