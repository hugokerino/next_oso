import torch

import lightning as L

import os
import sys
import pprint
from argparse import ArgumentParser

from embedding.dataset_embedding import embedding_datamodule
from embedding.decode_lightning_module import DecodeEmbedding
from src import metrics
from mains.main_utils import load_config, build_decoder


torch.serialization.add_safe_globals([metrics.FocalLoss])
torch.set_float32_matmul_precision("high")


def main(config, log_dir, save_preds_dir):
    ########################################
    # MODEL
    ########################################
    print("\nLOADING MODEL ...")
    sys.stdout.flush()

    model = config["model"]
    # Data
    datamodule = embedding_datamodule(
        data=config["data"],
        model=model,
        path_embeddings=config["path_embedding"],
        path_labels=config["path_label"],
        path_mapping_folder=config["path_mapping_folder"],
        fold_train=config["folds_train"],
        fold_val=config["folds_val"],
        fold_test=config["folds_test"],
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        preload=config["preload_data"],
        label_type=config["label_type"],
        mask=True,
        return_id=True,
    )
    # Model
    config["path_save_excel"] = f"{log_dir}/metrics.xlsx"
    decoder = build_decoder(model, config["decoder_type"], config["data"], dropout=0.1)

    ########################################
    # INFERENCE
    ########################################

    print("\nTESTING ...")
    sys.stdout.flush()
    checkpoint = os.listdir(os.path.join(log_dir, "Checkpoints/"))[0]
    print(f"Loading checkpoint: {checkpoint}")
    best_classifier = DecodeEmbedding.load_from_checkpoint(
        os.path.join(log_dir, "Checkpoints", checkpoint), decoder=decoder, config=config
    )

    best_classifier.eval()
    # Save per-sample predictions
    out_dir = os.path.join(save_preds_dir, f"preds_{config['model']}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_classifier.to(device)

    saved_count = 0
    skipped_count = 0
    with torch.no_grad():
        for batch_idx, (embeddings, _, ids) in enumerate(datamodule.test_dataloader()):
            # inputs: tuple of tensors (s2, dates, [mask], [geo]) with batch dim first
            embeddings = embeddings.to(device)
            batch_size = embeddings.shape[0]
            for i in range(batch_size):
                sample_id = str(ids[i]) if ids is not None else None
                if sample_id is None:
                    continue

                filename = os.path.join(
                    out_dir, f"pred_{config['model']}_{sample_id}.pt"
                )
                if os.path.exists(filename):
                    skipped_count += 1
                    continue

                # build per-sample inputs (each element has batch dim)
                sample_inputs = embeddings[i].unsqueeze(dim=0)
                # Forward single sample through the LightningModule
                y_logit = best_classifier.forward(sample_inputs)
                probas = torch.softmax(y_logit, dim=1)  # shape (1, C, H, W)

                probs_tensor = probas.squeeze(0).cpu()
                pred_tensor = probs_tensor.argmax(dim=0).to(torch.uint8)
                probs_tensor = torch.round(probs_tensor * 100).to(
                    torch.uint8
                )  # Scale to 0-100 and convert to uint8
                torch.save({"probs": probs_tensor, "pred": pred_tensor}, filename)
                saved_count += 1

    print(f"Saved {saved_count} new per-sample predictions to {out_dir}")
    if skipped_count:
        print(f"Skipped {skipped_count} existing files")


if __name__ == "__main__":
    print("\nPARSE ARGUMENTS ...")
    L.seed_everything(42, workers=True)
    parser = ArgumentParser()
    parser.add_argument(
        "--config_train", type=str, default="configs/FM_models/train_local.yaml"
    )
    parser.add_argument(
        "--config_model",
        type=str,
        required=True,
        help="Ex: configs/FM_models/alpha-earth.yaml",
    )
    parser.add_argument(
        "--path_dir_log",
        type=str,
        required=True,
        help="Chemin du dossier de logs, Ex:/home/hkerino/Documents/next_oso/logs/alpha-earth/2026-07-22_mlp_kfold",
    )
    parser.add_argument(
        "--path_save_preds",
        type=str,
        required=True,
        help="Chemin du dossier pour sauvergarder les prédictions, Ex:/home/hkerino/Documents/nextoso-dataset/preds",
    )
    args = parser.parse_args()

    config = load_config(args.config_train, args.config_model)
    pprint.pprint(config)
    main(config, args.path_dir_log, args.path_save_preds)
