import lightning as L
from lightning import Trainer

import torch
import sys
import pprint
from pathlib import Path
from argparse import ArgumentParser

import yaml

from embedding import dataset_embedding, decode_lightning_module

from mains.main_utils import get_precision, build_decoder

torch.set_float32_matmul_precision("high")


def main(config):
    # Paths
    model = config["model"]

    # Data
    # config["selected_fold"] holds the actual tuple/list of [train, val, test] for this job
    selected_fold = config["selected_fold"]

    datamodule = dataset_embedding.embedding_datamodule(
        dataset=config["dataset"],
        model=model,
        path_data=config["path_data"],
        fold_train=selected_fold[0],
        fold_val=selected_fold[1],
        fold_test=selected_fold[2],
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        preload=config["preload_data"],
        label_type=config["label_type"],
        label_type_test=config["label_type_test"],
        mask=True,
    )

    # Model
    print("\nTESTING ...")
    checkpoint_dir = (
        Path(config["path_dir_logs"])
        / f"fold_{config['fold_index'] + 1}"
        / "Checkpoints"
    )
    checkpoint = next(checkpoint_dir.glob("*.ckpt"))

    print(f"Loading checkpoint: {checkpoint}")
    sys.stdout.flush()

    log_fold = Path(config["path_dir_logs"]) / f"fold_{config['fold_index'] + 1}"
    config["path_save_excel"] = f"{log_fold}/metrics_testing.xlsx"
    config["path_save_raw_matrix"] = f"{log_fold}/raw_matrix_testing.pt"

    decoder = build_decoder(model, config["decoder_type"], config["dataset"], dropout=0)
    best_classifier = decode_lightning_module.DecodeEmbedding.load_from_checkpoint(
        checkpoint,
        decoder=decoder,
        config=config,
        names_class=datamodule.data_test.name_labels,
    )
    best_classifier.eval()

    # training
    trainer = Trainer(
        logger=None,
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=get_precision(),
        limit_test_batches=1.0,
    )

    trainer.test(best_classifier, datamodule=datamodule)


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing which embedding to use, "
        "ex: configs/FM_models/config_trex.yaml",
    )
    parser.add_argument(
        "--fold_index", type=int, required=True, help="Index of the fold to train on"
    )

    parser.add_argument(
        "--path_dir_logs",
        type=str,
        required=True,
        help="Path to trained directory. "
        "Should follow structure path_dir_logs/fold_*/Checkpoints/*.ckpt format, "
        "Ex:/home/hkerino/Documents/next_oso/logs/alpha-earth/2026-07-22_mlp_kfold",
    )
    args = parser.parse_args()

    # Open YAML file
    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
    config = {**train_cfg}

    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index
    config["path_dir_logs"] = args.path_dir_logs

    pprint.pprint(config)
    main(config)
