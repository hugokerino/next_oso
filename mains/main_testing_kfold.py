import torch
import lightning as L
import pandas
import pprint
from argparse import ArgumentParser
import sys
from pathlib import Path

from data_functions import dataset
from src import lightning_module, metrics
from mains.main_utils import load_config, build_model, select_criterion, get_precision

torch.serialization.add_safe_globals([metrics.FocalLoss])
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(
    False
)  # if using flash attention implementation batch_size cannot exceed 65535 (int16)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # no constraint on bacth size but more VRAM usage, should be less efficient


def main(config):
    print(f"FOLD {config['fold_index']}")
    sys.stdout.flush()

    criterion = select_criterion(config["criterion"])
    encoder, decoder, batch_size = build_model(config)
    config["batch_size"] = (
        batch_size  # Update batch size in config for data loading and lightning module
    )
    # config["selected_fold"] holds the actual tuple/list of [train, val, test] for this job
    selected_fold = config["selected_fold"]

    _, _, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_mnt=config[
            "path_mnt"
        ],  # if not None, add spatial vars to the decoding part
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=selected_fold[0],
        folds_val=selected_fold[1],
        folds_test=selected_fold[2],
        batch_size=config["batch_size"],
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=config["mask_need"],
        norm=True
        if config["model"] in ["LTAE", "UTAE", "ALISE", "ALISE_encoding"]
        else False,
        norm_alise=True if config["model"] == "ALISE_pretrained" else False,
    )

    trainer = L.Trainer(
        logger=None,
        log_every_n_steps=10,
        accelerator=config["accelerator"],
        devices=config["devices"],
        num_nodes=config["nodes"],
        strategy=config["strategy"],
        precision=get_precision(),
        limit_test_batches=1.0,
    )

    ########################################
    # TESTING
    ########################################
    print("\nTESTING ...")
    checkpoint_dir = (
        Path(config["path_dir_checkpoints"])
        / f"fold_{config['fold_index'] + 1}"
        / "Checkpoints"
    )
    checkpoint = next(checkpoint_dir.glob("*.ckpt"))

    print(f"Loading checkpoint: {checkpoint}")
    sys.stdout.flush()

    best_classifier = lightning_module.classifier.load_from_checkpoint(
        checkpoint_path=checkpoint,
        weights_only=True,
        encoder=encoder,
        decoder=decoder,
        loss_fn=criterion,
        batch_size=batch_size,
        name_class=pandas.read_csv(config["path_oso_classes"]).get("name_class").values,
        path_raw_matrix=f"{config['path_dir_checkpoints']}/fold_{config['fold_index'] + 1}/raw_matrix.pt",
    )
    
    best_classifier.eval()

    trainer.test(model=best_classifier, dataloaders=dl_test)


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()
    parser.add_argument(
        "--config_train", type=str, default="configs/models/train_local.yaml"
    )
    parser.add_argument(
        "--config_model", type=str, required=True, help="Ex: configs/models/LTAE.yaml"
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        required=True,
        help="Index of the fold to run (0-based, e.g., 0 for first fold)",
    )

    parser.add_argument(
        "--path_dir_checkpoints",
        type=str,
        required=True,
        help="Chemin du dossier de logs, Ex:/home/hkerino/Documents/next_oso/logs/light_training/LTAE_2026-06-01",
    )

    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)

    config["path_dir_checkpoints"] = args.path_dir_checkpoints
    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index

    pprint.pprint(config)

    main(config)
