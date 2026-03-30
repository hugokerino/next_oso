import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.plugins.environments import SLURMEnvironment

import numpy as np
import signal
import ast
import os
from argparse import ArgumentParser
import sys
from datetime import datetime
import pprint

from data_functions import dataset
from src import LTAE, UTAE, utils_model, lightning_module, metrics
from Alise import ALISE, sse, transformer, temporal_proj

# Add FocalLoss for loading best model from checkpoint
torch.serialization.add_safe_globals([metrics.FocalLoss])


def parse_args_to_lists(config):
    # Convert str representing a list to a list of int
    for k, v in config.items():
        if "folds" in k:
            try:
                # Utilise ast.literal_eval pour convertir la chaîne en liste de manière sûre
                config[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                # Si ast.literal_eval échoue, essaie une conversion manuelle
                v = v.strip("[]")  # Supprime les crochets
                v = v.replace(" ", "")  # Supprime les espaces
                config[k] = [
                    int(x) for x in v.split(",") if x
                ]  # Convertit en liste d'entiers
    return config


def main(config):
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
        path_s2=config["path_s2"],
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=config["folds_train"],
        folds_val=config["folds_val"],
        folds_test=config["folds_test"],
        batch_size=config["batch_size"],
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        norm=True,
        remove_invalid_temp=True,
        shuffle=True,
    )

    #
    ########################################
    # # # MODEL # # #
    ########################################
    #
    print("\nLOADING MODEL ...")
    sys.stdout.flush()

    # Loss function
    alpha_weights = torch.from_numpy(
        np.load("./data_functions/alpha_weights.npy")
    ).float()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0, weight=alpha_weights)
    # criterion = metrics.FocalLoss(alpha=alpha_weights, gamma=2, ignore_index=0)  #
    learning_rate = 1e-3

    # Model
    if config["model"] == "LTAE":
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
        mask_need = True

        # Decoder
        MLP_Decoder = utils_model.MLPDecoder(mlp=[128, 64, 25], dropout=0.2)

        encoder = LTAE_encoder
        decoder = MLP_Decoder

    if config["model"] == "UTAE":
        # Encoder
        UTAE_encoder = UTAE.UTAE(
            input_dim=10,
            encoder_widths=[32, 64, 64, 128],
            decoder_widths=[32, 64, 64, 128],
            out_conv=[32, 25],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=8,
            encoder=True,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
        )
        mask_need = False

        # Decoder
        MLP_Decoder = utils_model.MLPDecoder(mlp=[32, 25], dropout=0.2)

        encoder = UTAE_encoder
        decoder = MLP_Decoder

    if config["model"] == "ALISE":
        # Encoder
        embedding_dim = 64
        ALISE_encoder = ALISE.ALISE(
            patch_embedding=sse.PatchEmbedding(
                sse=sse.Unet(
                    inplanes=10,
                    planes=embedding_dim,
                    encoder_widths=[64, 64, 64, 128],
                    decoder_widths=[32, 32, 64, 128],
                    encoder_norm="group",
                    padding_mode="reflect",
                    decoding_norm="group",
                    return_maps=False,
                    str_conv_k=4,
                    str_conv_s=2,
                    str_conv_p=1,
                    skip_conv_norm="group",
                ),
                tpe=sse.PositionalEncoder(d=embedding_dim),
            ),
            temporal_encoder=transformer.Transformer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=embedding_dim * 2,
                dropout=0.2,
                batch_first=True,
                num_layers=3,
            ),
            temporal_projector=temporal_proj.LearnedQMultiHeadAttention(
                n_head=8, d_k=8, d_in=embedding_dim, n_q=1
            ),
        )
        mask_need = False

        # Decoder
        MLP_Decoder = utils_model.MLPDecoder(
            mlp=[64, 32, 25], dropout=0.2
        )

        encoder = ALISE_encoder
        decoder = MLP_Decoder

    # Classifier
    classifier = lightning_module.classifier(
        encoder,
        decoder,
        batch_size=config["batch_size"],
        loss_fn=criterion,
        mask_need=mask_need,
        lr=learning_rate,
    )

    #
    ########################################
    # # # TRAINING # # #
    ########################################
    #
    print("\nTRAINING ...")
    sys.stdout.flush()

    # Training params
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            config["path_logs"], f"{config['model']}_{date}/Checkpoints/"
        ),
        save_weights_only=True,
        monitor="val_fscore",
        mode="max",
        save_top_k=1,
    )

    tb_logger = TensorBoardLogger(
        config["path_logs"], name=f"{config['model']}_{date}", default_hp_metric=False
    )

    callbacks = EarlyStopping(monitor="val_fscore", mode="max", patience=10)
    profiler = SimpleProfiler(
        dirpath=os.path.join(config["path_logs"], f"{config['model']}_{date}"),
        filename="profiler",
    )
    list_callback = [checkpoint_callback, callbacks]
    limit_batches = 1.0  # full

    # Testing config, desactivate loggers, callback & profiler for testing, limit number of batches
    if config["test_model"]:
        tb_logger = None
        list_callback = None
        profiler = None
        limit_batches = 2

    plugin = None
    if config["running_on_trex_gpu"]:
        plugin = [SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)]

    # Trainer
    trainer = L.Trainer(
        logger=tb_logger,
        log_every_n_steps=10,
        callbacks=list_callback,
        max_epochs=config["epochs_max"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        num_nodes=config["nodes"],
        strategy=config["strategy"],
        profiler=profiler,
        num_sanity_val_steps=limit_batches,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        precision="16-mixed",
        plugins=plugin,
    )

    # Training
    trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_val)

    # Testing
    print("\nTesting ... ")
    sys.stdout.flush()

    best_classifier = classifier
    if not config["test_model"]:
        # Get best params and test with them
        best_model_path = checkpoint_callback.best_model_path
        best_classifier = lightning_module.classifier.load_from_checkpoint(
            best_model_path,
            weights_only=True,
            encoder=encoder,
            decoder=decoder,
            loss_fn=criterion,
            batch_size=config["batch_size"],
        )

    trainer.test(model=best_classifier, dataloaders=dl_test)


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ... ")
    parser = ArgumentParser()

    # Paths
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
        default="/home/hkerino/Documents/next_oso/logs/light_training",
    )

    # Model
    parser.add_argument("--model", type=str)

    # Data
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--label_type", type=str, default="max_class")
    parser.add_argument("--num_class", type=int, default=25)
    parser.add_argument(
        "--folds_train", type=str, default="[1, 2, 3]"
    )  # To convert in list
    parser.add_argument("--folds_val", type=str, default="[4]")  # To convert in list
    parser.add_argument("--folds_test", type=str, default="[5]")  # To convert in list

    # Training
    parser.add_argument("--epochs_max", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_model", type=int, default=0)  # true/false
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--running_on_trex_gpu", type=int, default=0)  # true/false

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    config = vars(parser.parse_args())
    config = parse_args_to_lists(config)
    pprint.pprint(config)

    main(config)
