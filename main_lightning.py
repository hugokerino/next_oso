import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from pytorch_lightning.plugins.environments import SLURMEnvironment
import numpy as np
import signal
import os
from argparse import ArgumentParser
import sys
from datetime import datetime

from data_functions import dataset
from src import LTAE, utils_model, lightning_module, metrics

# Add FocalLoss for loading best model from checkpoint
torch.serialization.add_safe_globals([metrics.FocalLoss])


def main(args):
    #
    ########################################
    # # # ARGUMENTS, PATHS & DATA # # #
    ########################################
    #
    date = datetime.now().date().strftime("%Y-%m-%d")

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
    print("\nLOADING MODEL ...")
    sys.stdout.flush()
    # Encoder
    LTAE_encoder = LTAE.LTAE2d(
        in_channels=10,
        n_head=16,
        d_k=8,
        d_model=128,
        T=1000,
        return_att=False,
        positional_encoding=True,
    )

    # Decoder
    MLP_Decoder = utils_model.MLPDecoder(mlp=[128, 64, 25], dropout=0.2)

    # Classifier
    # criterion_ce = F.CrossEntropyLoss(ignore_index=0)
    alpha_weights = torch.from_numpy(np.load("./data_functions/alpha_weights.npy"))
    criterion_fl = metrics.FocalLoss(alpha=alpha_weights, gamma=2, ignore_index=0)
    classifier = lightning_module.classifier(
        LTAE_encoder,
        MLP_Decoder,
        batch_size=args.batch_size,
        loss_fn=criterion_fl,
        lr=1e-2,
    )

    #
    ########################################
    # # # TRAINING # # #
    ########################################
    #
    print("\nTRAINING ...")
    sys.stdout.flush()

    # Test models training over 3 batchs on each train,validation & testing datasets
    if args.test_model == "True":
        trainer_test = L.Trainer(fast_dev_run=3, num_sanity_val_steps=None)
        trainer_test.fit(
            model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_val
        )
        trainer_test.test(model=classifier, dataloaders=dl_test)

    # Real training params
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.path_logs, f"LTAE_{date}/Checkpoints/"),
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
     
    tb_logger = TensorBoardLogger(
        	args.path_logs, name=f"LTAE_{date}", default_hp_metric=False
    		)

    callbacks = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    profiler = SimpleProfiler(
        dirpath=os.path.join(args.path_logs, f"LTAE_{date}"), filename="profiler"
    )
    plugin=SLURMEnvironment(requeue_signal=signal.SIGHUP)
    
    trainer = L.Trainer(
        logger=tb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, callbacks],
        max_epochs=args.epochs_max,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        strategy=args.strategy,
        profiler=profiler,
        num_sanity_val_steps=1,
        #limit_train_batches=2,
        #limit_val_batches=2,
        #limit_test_batches=2,
        precision="16-mixed",
        #plugins=[plugin],
    )

    # Training
    trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_val)

    # Get best params and test with them
    best_model_path = checkpoint_callback.best_model_path
    best_classifier = lightning_module.classifier.load_from_checkpoint(
        best_model_path,
        weights_only=True,
        encoder=LTAE_encoder,
        decoder=MLP_Decoder,
        loss_fn=criterion_fl,
        batch_size=args.batch_size,
    )
    print("\nTesting ... ")
    sys.stdout.flush()
    trainer.test(model=best_classifier, dataloaders=dl_test)


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ... ")
    parser = ArgumentParser()

    # Hyperparameters for the model
    parser.add_argument(
        "--path_s2",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
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
        default="/home/hkerino/Documents/models_on_nextoso/logs/light_training",
    )
    parser.add_argument("--epochs_max", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--num_class", type=int, default=25)
    parser.add_argument("--folds_train", type=list, default=[1,2,3])
    parser.add_argument("--folds_val", type=list, default=[4])
    parser.add_argument("--folds_test", type=list, default=[5])
    parser.add_argument("--test_model", type=str, default="False")
    parser.add_argument("--logger", type=str,default="True")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--nodes", type=int,default=1)
    parser.add_argument("--strategy", type=str, default="auto")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    main(args)
