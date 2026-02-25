import torch
import numpy as np
import pandas
import sys
import os
import json
from datetime import datetime
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter

from data_functions import dataset
from src import pytorch_training, metrics
from src.pytorch_training import train, val

#
########################################
# # # ARGUMENTS, PATHS & DATA # # #
########################################
#
print("\nPARSE ARGUMENTS ... ")
parser = ArgumentParser()

# Hyperparameters for the model
parser.add_argument(
    "--path_s2", type=str, default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2"
)
parser.add_argument(
    "--path_label",
    type=str,
    default="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
)
parser.add_argument(
    "--path_oso_classes",
    type=str,
    default="/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
)
parser.add_argument(
    "--path_logs",
    type=str,
    default="/home/hkerino/Documents/models_on_nextoso/logs/torch_training",
)
parser.add_argument("--epochs_max", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_classes", type=int, default=25)
parser.add_argument("--num_worker", type=int, default=4)
parser.add_argument("--num_class", type=int, default=25)
parser.add_argument("--folds_train", type=list, default=[1, 2, 3])
parser.add_argument("--folds_val", type=list, default=[4])
parser.add_argument("--folds_test", type=list, default=[5])
parser.add_argument("--test_model", type=str, default="False")

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()
date = datetime.now().date().strftime("%Y-%m-%d")
PATH_TO_SAVE = os.path.join(args.path_logs, date)
num_classes = args.num_classes

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
else:
    print("Périphérique utilisé : cpu")

print(f"Périphérique PyTorch : {device}")

# Load DataLoaders
print("\nLOADING DATA ...")
sys.stdout.flush()
dl_train, dl_val, dl_test = dataset.get_loaders(
    path_s2=args.path_s2,
    path_label=args.path_label,
    path_oso_class=args.path_oso_classes,
    folds_train=args.folds_train,
    folds_val=args.folds_val,
    folds_test=args.folds_test,
    batch_size=args.batch_size,
    num_worker=args.num_worker,
    norm=True,
    shuffle=True,
)

#
########################################
# # # MODEL # # #
########################################
#
# Encoder
model_type = "LTAE"
model = pytorch_training.Classifier_LTAE(
    in_channels=10,
    n_head=8,
    d_k=8,
    d_model=128,
    T=1000,
    return_att=False,
    return_encoder=False,
    positional_encoding=True,
    mlp=[128, 64, num_classes],
    dropout=0.2,
)
mask_needed = True
model = model.to(device)


#
########################################
# # # TRAINING # # #
########################################
#
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epoch = args.epochs_max


print("\nTRAINING")
# Tensorboard
writer = SummaryWriter(PATH_TO_SAVE)

# Loop training
best_mIoU = 0
trainlog = {}
loss_train_list = []
loss_val_list = []

# First forward with untrained network, epoch "0"
print("First evaluation w/o model training")
sys.stdout.flush()
model.eval()
val_metrics, loss_val = val(
    model,
    dl_val,
    criterion,
    device,
    0,
    mask_needed=mask_needed,
    writer=writer,
    mode="validation",
    num_classes=num_classes,
)

loss_val_list.append(loss_val)
if writer is not None:
    writer.add_scalar("Val/Loss", val_metrics["Loss"], 0)
    writer.add_scalar("Val/Accuracy", val_metrics["Accuracy"], 0)

for e in range(1, epoch + 1):
    print(f"Epoch {e}/{epoch}")
    print("Training . . .")
    sys.stdout.flush()
    model.train()
    train_metrics, loss_train = train(
        model,
        dl_train,
        criterion,
        optimizer,
        device,
        e,
        mask_needed=mask_needed,
        writer=writer,
        num_classes=num_classes,
        display_step=True,
    )

    print("Validation . . .")
    sys.stdout.flush()

    model.eval()
    val_metrics, loss_val = val(
        model,
        dl_val,
        criterion,
        device,
        e,
        mask_needed=mask_needed,
        writer=writer,
        mode="validation",
        num_classes=num_classes,
    )

    if val_metrics["mIoU"] >= best_mIoU:
        best_mIoU = val_metrics["mIoU"]
        torch.save(
            model.state_dict(),
            os.path.join(PATH_TO_SAVE, "best_params_model_" + model_type + ".pth"),
        )

    trainlog[epoch] = {**train_metrics, **val_metrics}
    loss_train_list.extend(loss_train)
    loss_val_list.append(loss_val)


model.load_state_dict(
    torch.load(os.path.join(PATH_TO_SAVE, "best_params_model_" + model_type + ".pth"))
)

print("\nEvaluation on data test")
sys.stdout.flush()
model.eval()
test_metrics, test_confusion_matrix = val(
    model,
    dl_test,
    criterion,
    device,
    mask_needed=mask_needed,
    mode="test",
    num_classes=num_classes,
)


#
########################################
# # # SAVE # # #
########################################
#
print("\nSAVE RESULTS")
sys.stdout.flush()

# Tensorboard
class_labels = pandas.read_csv(args.path_oso_classes).get("name_class")

plot_test_confusion_matrix = metrics.return_plot_confusion_matrix(
    test_confusion_matrix, class_labels
)
fig, mean_scores = metrics.return_plot_f_score(test_confusion_matrix, class_labels)
test_metrics.update(mean_scores)

if writer is not None:
    writer.add_figure(
        "Confusion Matrix",
        plot_test_confusion_matrix,
    )
    writer.add_figure(
        "Precision, Recall, F-Score",
        fig,
    )

    table_str = metrics.create_markdown_table(test_metrics)
    writer.add_text("Test_Metrics", table_str)

    writer.close()


# .json / .npy
with open(os.path.join(PATH_TO_SAVE, "loss_train.json"), "w") as f:
    json.dump(loss_train_list, f)

with open(os.path.join(PATH_TO_SAVE, "loss_val.json"), "w") as f:
    json.dump(loss_val_list, f)

with open(os.path.join(PATH_TO_SAVE, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f)

np.save(os.path.join(PATH_TO_SAVE, "test_confusion_matrix.npy"), test_confusion_matrix)
