import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score

from argparse import ArgumentParser
import sys
from datetime import datetime

from data_functions import dataset
from src import LTAE, utils_model, lightning_module


#
########################################
# # # ARGUMENTS, PATHS & DATA # # #
########################################
#
print("\nPARSE ARGUMENTS ")
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
parser.add_argument("--epochs_max", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_worker", type=int, default=4)
parser.add_argument("--num_class", type=int, default=25)
parser.add_argument("--folds_train", type=list, default=[1])
parser.add_argument("--folds_val", type=list, default=[4])
parser.add_argument("--folds_test", type=list, default=[5])
parser.add_argument("--test_model", type=bool, default=True)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()
date = datetime.now().date().strftime("%Y-%m-%d")

# Load DataLoaders
print("\nLOADING DATA ...")
sys.stdout.flush()
dl_train, dl_val, dl_test = dataset.get_loaders(
    path_s2=args.path_s2,
    path_label=args.path_label,
    folds_train=args.folds_train,
    folds_val=args.folds_val,
    folds_test=args.folds_test,
    batch_size=args.batch_size,
    num_worker=args.num_worker,
    norm=True,
    shuffle=True,
)

# Select device
print("\nDEVICE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
else:
    print("Périphérique utilisé : cpu")

print(f"Périphérique PyTorch : {device}")
sys.stdout.flush()


#
########################################
# # # MODEL # # #
########################################
#
print("\nLOADING MODEL ...")
sys.stdout.flush()
# Encoder
LTAE_encoder = LTAE.LTAE2d(
    in_channels=10,
    n_head=16,
    d_k=16,
    d_model=128,
    T=1000,
    return_att=False,
    positional_encoding=True,
)

# Decoder
MLP_Decoder = utils_model.MLPDecoder(mlp=[128, 64, 25], dropout=0.2)

# Classifier
classifier = lightning_module.classifier(
    LTAE_encoder, MLP_Decoder, batch_size=args.batch_size
)


#
########################################
# # # TRAINING # # #
########################################
#
print("\nTRAINING ...")
sys.stdout.flush()

# Test models training over 3 batchs on each train,validation & testing datasets
if args.test_model:
    trainer_test = L.Trainer(fast_dev_run=3)
    trainer_test.fit(
        model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_val
    )
    trainer_test.test(model=classifier, dataloaders=dl_test)

# Real training params
checkpoint_callback = ModelCheckpoint(
    dirpath=f"/home/hkerino/Documents/models_on_nextoso/logs/LTAE_{date}/Checkpoints/",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)
tb_logger = TensorBoardLogger("logs/", name=f"LTAE_{date}")
callbacks = EarlyStopping(monitor="val_loss", mode="min", patience=3)

trainer = L.Trainer(
    logger=tb_logger,
    callbacks=[checkpoint_callback, callbacks],
    max_epochs=args.epochs_max,
    profiler="simple",
    num_sanity_val_steps=2,
)

# Training
trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_val)

# Get best params and test with them
best_model_path = checkpoint_callback.best_model_path
best_classifier = classifier.load_from_checkpoint(best_model_path)
trainer.test(model=best_classifier, dataloaders=dl_test)
