import os
import yaml
import pandas
import torch
import pickle
import pprint
from argparse import ArgumentParser
from collections import defaultdict

from src import lightning_module
from data_functions.dataset import next_oso_Dataset

from embedding import decode_lightning_module
from embedding.dataset_embedding import embedding_dataset

from mains.main_utils import build_decoder, build_model, select_criterion
from supervised_models.utils_model import MLP_Decoder

FOLDS = {1: 5, 2: 1, 3: 2, 4: 3, 5: 4}


def main(args):
    dict_results = defaultdict(lambda: defaultdict(int))

    kfold = args.kfold

    # ALISE
    with open(args.config_alise, "r") as f:
        model_cfg = yaml.safe_load(f)
    config = {**model_cfg}
    config["path_mnt"] = args.path_mnt
    config["decoder_type"] = "mlp"
    config["batch_size"] = 1

    encoder, decoder, batch_size = build_model(config)
    # ALISE classifier
    alise_ckpt = os.listdir(f"{args.alise_checkpoint}/fold_{kfold}/Checkpoints/")[0]
    ALISE = lightning_module.classifier.load_from_checkpoint(
        os.path.join(args.alise_checkpoint, f"fold_{kfold}/Checkpoints/", alise_ckpt),
        weights_only=True,
        encoder=encoder,
        decoder=decoder,
        loss_fn=select_criterion("cross_entropy"),
        batch_size=batch_size,
        name_class=pandas.read_csv(args.path_oso_classes).get("name_class").values,
    )

    # ALPHA-EARTH
    decoder_ae = build_decoder(
        "alpha-earth", config["decoder_type"], "next_oso", dropout=0
    )
    # ae classifier
    ae_ckpt = os.listdir(f"{args.ae_checkpoint}/fold_{kfold}/Checkpoints/")[0]
    ALPHA_EARTH = decode_lightning_module.DecodeEmbedding.load_from_checkpoint(
        os.path.join(args.ae_checkpoint, f"fold_{kfold}/Checkpoints/", ae_ckpt),
        decoder=decoder_ae,
        config={
            "data": "next_oso",
            "lr": 0.001,
            "max_epochs": 300,
            "path_name_class": args.path_oso_classes,
            "path_save_excel": None,
        },
    )

    # MALICE
    decoder_malice = MLP_Decoder(
        layers=[640, 320, 64, 25],
        dropout=0,
    )
    # malice classifier
    malice_ckpt = os.listdir(f"{args.malice_checkpoint}/fold_{kfold}/Checkpoints/")[0]
    MALICE = decode_lightning_module.DecodeEmbedding.load_from_checkpoint(
        os.path.join(args.malice_checkpoint, f"fold_{kfold}/Checkpoints/", malice_ckpt),
        decoder=decoder_malice,
        config={
            "data": "next_oso",
            "lr": 0.001,
            "max_epochs": 300,
            "path_name_class": args.path_oso_classes,
            "path_save_excel": None,
        },
    )

    # Load test data
    data_alise = next_oso_Dataset(
        path_s2=args.path_s2,
        path_label=args.path_label,
        path_oso_class=args.path_oso_classes,
        path_mnt=config["path_mnt"],
        folds=[FOLDS[kfold]],
        mask_needed=True,
    )
    data_ae = embedding_dataset(
        path_embeddings=args.path_embeddings,
        path_labels=args.path_label,
        path_mapping_folder=args.path_mapping_folder,
        fold=[FOLDS[kfold]],
        model="alpha-earth",
        return_id=True,
    )
    data_malice = embedding_dataset(
        path_embeddings=args.path_embeddings,
        path_labels=args.path_label,
        path_mapping_folder=args.path_mapping_folder,
        fold=[FOLDS[kfold]],
        model="malice",
        return_id=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nModel on device: {device}")

    ALPHA_EARTH.to(device)
    ALISE.to(device)
    MALICE.to(device)

    ALPHA_EARTH.eval()
    ALISE.eval()
    MALICE.eval()

    for sample in range(0, data_ae.__len__()):
        # Get sample
        embedding_ae, label_ae, id = data_ae.__getitem__(sample)
        embedding_malice, label_malice = data_malice.__getitem__(id)
        (s2, date, mask, geo), label_alise = data_alise.__getitem__(id)

        # to device
        embedding_ae = embedding_ae.to(device)
        label_ae = label_ae.to(device)

        embedding_malice = embedding_malice.to(device)
        label_malice = label_malice.to(device)

        s2 = s2.to(device)
        date = date.to(device)
        mask = mask.to(device)
        geo = geo.to(device)
        label_alise = label_alise.to(device)

        # Inference
        with torch.no_grad():
            preds_ae = (
                torch.softmax(ALPHA_EARTH(embedding_ae.unsqueeze(0)), dim=1)
                .argmax(dim=1)
                .squeeze(dim=0)
            )  # h w
            preds_malice = (
                torch.softmax(MALICE(embedding_malice.unsqueeze(0)), dim=1)
                .argmax(dim=1)
                .squeeze(dim=0)
            )  # h w
            preds_alise = (
                torch.softmax(
                    ALISE._forward(
                        (
                            (s2.unsqueeze(0), date.unsqueeze(0), geo.unsqueeze(0)),
                            label_alise.unsqueeze(0),
                        )
                    ),
                    dim=1,
                )
                .argmax(dim=1)
                .squeeze(dim=0)
            )
            # h w

        lenght = date.shape[0]
        mask = mask.int()

        all_invalid = (mask == 1).all(dim=0)  # (H, W)
        valid_pixels = ~all_invalid  # (H, W)

        valid_labels  = (label_alise != 0) & (label_ae != 0) & (label_malice != 0)  # (H, W)
        # ── Combiner les deux masques ──
        valid_pixels  = valid_pixels & valid_labels

        # ── Nombre de timesteps valides PAR PIXEL ──
        valid_timesteps = lenght - mask.sum(
            dim=0
        )  # (H, W) — valeur différente pour chaque pixel
        # ex:
        # pixel (0,0) → 10 timesteps valides
        # pixel (0,1) → 7 timesteps valides
        # pixel (1,5) → 15 timesteps valides

        # Garder seulement les pixels valides → shape (N_valid,)
        valid_timesteps_flat = valid_timesteps[
            valid_pixels
        ]  # (N_valid,) — un vt par pixel valide
        correct_alise_flat = (preds_alise == label_alise)[valid_pixels]
        correct_ae_flat = (preds_ae == label_ae)[valid_pixels]
        correct_malice_flat = (preds_malice == label_malice)[valid_pixels]

        # ── Remplir le dict — une valeur de vt par pixel ──
        for vt in valid_timesteps_flat.unique():
            vt_mask = (
                valid_timesteps_flat == vt
            )  # pixels ayant exactement ce nb de timesteps valides
            vt_key = vt.item()

            dict_results[vt_key]["nb"] += vt_mask.sum().item()
            dict_results[vt_key]["ALISE"] += correct_alise_flat[vt_mask].sum().item()
            dict_results[vt_key]["alpha-earth"] += correct_ae_flat[vt_mask].sum().item()
            dict_results[vt_key]["MALICE"] += correct_malice_flat[vt_mask].sum().item()

    dict_results = {k: dict(v) for k, v in dict_results.items()}

    with open(
        args.path_output_dict + f"/dict_results_{kfold}.pkl", "wb"
    ) as f:  # "wb" = write binary
        pickle.dump(dict_results, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    ### ALISE
    parser.add_argument("--config_alise", type=str, default="configs/models/ALISE.yaml")
    parser.add_argument(
        "--alise_checkpoint",
        type=str,
        default="/home/hkerino/Documents/next_oso/logs/light_training/ALISE_kfold_spatial_07-17",
    )
    parser.add_argument(
        "--path_mnt",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/mnt",
    )
    parser.add_argument(
        "--path_oso_classes",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
    )
    parser.add_argument(
        "--path_s2",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    )

    ### ALPHA-EARTH / MALICE
    parser.add_argument(
        "--ae_checkpoint",
        type=str,
        default="/home/hkerino/Documents/next_oso/logs/alpha-earth/2026-07-22_mlp_kfold",
    )
    parser.add_argument(
        "--malice_checkpoint",
        type=str,
        default="/home/hkerino/Documents/next_oso/logs/malice/malice_kfold_mlp_22-07",
    )

    parser.add_argument(
        "--path_embeddings",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset",
    )
    parser.add_argument(
        "--path_mapping_folder",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    )

    ## General
    parser.add_argument(
        "--path_label",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    )
    parser.add_argument(
        "--path_output_dict",
        type=str,
        default="/home/hkerino/Documents/next_oso/logs/todel/test/",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    pprint.pprint(vars(args))

    main(args)
