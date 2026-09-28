import torch
import lightning as L

import pandas
import os
import sys
import pprint
from argparse import ArgumentParser
from data_functions import dataset
from src import lightning_module, metrics
from mains.main_utils import load_config, build_model, select_criterion


torch.serialization.add_safe_globals([metrics.FocalLoss])
torch.set_float32_matmul_precision("high")


def main(config, log_dir, save_preds_dir):
    ########################################
    # MODEL
    ########################################
    print(f"FOLD {config['fold_index']}")

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
    selected_fold = config["selected_fold"]

    _, _, dl_test = dataset.get_loaders(
        path_s2=config["path_s2"],
        path_mnt=config["path_mnt"],
        path_label=config["path_label"],
        path_oso_class=config["path_oso_classes"],
        folds_train=selected_fold[0],
        folds_val=selected_fold[1],
        folds_test=selected_fold[2],
        batch_size=batch_size,
        num_worker=config["num_worker"],
        label_type=config["label_type"],
        mask_needed=config["mask_need"],
        norm=True
        if config["model"] in ["LTAE", "UTAE", "ALISE", "ALISE_encoding"]
        else False,
        norm_alise=True if config["model"] == "ALISE_pretrained" else False,
        sequence_packing=config["sequence_packing"],
        remove_invalid_temp=True,
        shuffle=True,
        pass_id=True,
    )

    ########################################
    # INFERENCE
    ########################################
    print("\nTESTING ...")
    sys.stdout.flush()
    checkpoint = os.listdir(
        os.path.join(log_dir, f"fold_{config['fold_index'] + 1}/Checkpoints/")
    )[0]
    print(f"Loading checkpoint: {checkpoint}")

    best_classifier = lightning_module.classifier.load_from_checkpoint(
        os.path.join(log_dir, f"fold_{config['fold_index']+1}/Checkpoints", checkpoint),
        weights_only=True,
        encoder=encoder,
        decoder=decoder,
        loss_fn=criterion,
        batch_size=batch_size,
        name_class=pandas.read_csv(config["path_oso_classes"]).get("name_class").values,
    )

    best_classifier.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_classifier.to(device)

    # Save per-sample predictions
    saved_count = 0
    skipped_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, ids) in enumerate(dl_test):
            # inputs: tuple of tensors (s2, dates, [mask], [geo]) with batch dim first
            inputs = tuple(t.to(device) for t in inputs)

            batch_size = inputs[0].shape[0]
            for i in range(batch_size):
                sample_id = str(ids[i]) if ids is not None else None
                if sample_id is None:
                    continue

                filename = os.path.join(save_preds_dir, f"pred_{sample_id}.pt")
                if os.path.exists(filename):
                    skipped_count += 1
                    continue

                # build per-sample inputs (each element has batch dim)
                sample_inputs = [t[i].unsqueeze(dim=0) for t in inputs]

                # Forward single sample through the LightningModule
                y_logit = best_classifier._forward((sample_inputs, _))
                probas = torch.softmax(y_logit, dim=1)  # shape (1, C, H, W)

                probs_tensor = probas.squeeze(0).cpu()
                pred_tensor = probs_tensor.argmax(dim=0).to(torch.uint8)
                probs_tensor = torch.round(probs_tensor * 100).to(
                    torch.uint8
                )  # Scale to 0-100 and convert to uint8
                torch.save({"probs": probs_tensor, "pred": pred_tensor}, filename)
                saved_count += 1

    print(f"Saved {saved_count} new per-sample predictions to {save_preds_dir}")
    if skipped_count:
        print(f"Skipped {skipped_count} existing files")


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()
    parser.add_argument(
        "--config_train", type=str, default="configs/models/train_local.yaml"
    )
    parser.add_argument(
        "--config_model", type=str, required=True, help="Ex: configs/models/UTAE.yaml"
    )
    parser.add_argument(
        "--path_dir_log",
        type=str,
        required=True,
        help="Chemin du dossier de logs, Ex:/home/hkerino/Documents/next_oso/logs/light_training/LTAE_2026-06-01",
    )
    parser.add_argument(
        "--path_save_preds",
        type=str,
        required=True,
        help="Chemin du dossier pour sauvergarder les prédictions, Ex:/home/hkerino/Documents/nextoso-dataset/preds",
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        required=True,
        help="Index of the fold to run (0-based, e.g., 0 for first fold)",
    )

    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)

    config["path_dir_log"] = args.path_dir_log
    config["selected_fold"] = config["folds"][args.fold_index]
    config["fold_index"] = args.fold_index

    pprint.pprint(config)
    main(config, args.path_dir_log, args.path_save_preds)
