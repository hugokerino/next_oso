import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

import pandas
import os
import sys
import pprint
from argparse import ArgumentParser
from data_functions import dataset
from src import lightning_module, metrics
from mains.main_utils import load_config, build_model, select_criterion, get_precision


torch.serialization.add_safe_globals([metrics.FocalLoss])
torch.set_float32_matmul_precision("high")


def main(config, log_dir):

    ########################################
    # MODEL
    ########################################
    print("\nLOADING MODEL ...")
    sys.stdout.flush()

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    criterion = select_criterion(config["criterion"])
    encoder, decoder, batch_size = build_model(config)
    config["batch_size"] = (
        batch_size  # Update batch size in config for data loading and lightning module
    )

    ########################################
    # DATA
    ########################################
    print("\nLOADING DATA ...")
    sys.stdout.flush()

    _, _, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_mnt=config["path_mnt"],
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=config["folds_train"],
        folds_val=config["folds_val"],
        folds_test=config["folds_test"],
        batch_size=batch_size,
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=config["mask_need"],
        use_spatial_coords=config["use_spatial_coords"],
        norm=True
        if config["model"] in ["LTAE", "UTAE", "ALISE", "ALISE_encoding"]
        else False,
        norm_alise=True if config["model"] == "ALISE_pretrained" else False,
        sequence_packing=config["sequence_packing"],
        remove_invalid_temp=True,
        shuffle=True,
    )

    tb_logger = TensorBoardLogger(
        log_dir, name="tensorboards", version=0, default_hp_metric=False
    )

    trainer = L.Trainer(
        logger=tb_logger,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        max_epochs=config["epochs_max"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        num_nodes=config["nodes"],
        strategy=config["strategy"],
        precision=get_precision(),
    )
    ########################################
    # TESTING
    ########################################
    print("\nTESTING ...")
    sys.stdout.flush()
    checkpoint = os.listdir(os.path.join(log_dir, "Checkpoints/"))[
        0
    ]  # Assuming only one checkpoint is saved
    print(f"Loading checkpoint: {checkpoint}")
    best_classifier = lightning_module.classifier.load_from_checkpoint(
        os.path.join(log_dir, "Checkpoints", checkpoint),
        weights_only=True,
        encoder=encoder,
        decoder=decoder,
        loss_fn=criterion,
        batch_size=batch_size,
        use_spatial_coords=config["use_spatial_coords"],
        name_class=pandas.read_csv(config["path_oso_classes"]).get("name_class").values,
    )

    trainer.test(model=best_classifier, dataloaders=dl_test)


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()
    parser.add_argument("--config_train", type=str, default="configs/train_local.yaml")
    parser.add_argument(
        "--config_model", type=str, required=True, help="Ex: configs/models/UTAE.yaml"
    )
    parser.add_argument(
        "--path_dir_log",
        type=str,
        required=True,
        help="Chemin du dossier de logs, Ex:/home/hkerino/Documents/next_oso/logs/light_training/LTAE_2026-06-01",
    )
    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)
    pprint.pprint(config)
    main(config, args.path_dir_log)
