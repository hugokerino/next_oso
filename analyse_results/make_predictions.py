import torch
import os
from argparse import ArgumentParser

from embedding import dataset_embedding, decode_lightning_module
from mains.main_utils import build_model, build_decoder, load_config
from data_functions import dataset
import pprint
from pathlib import Path

PATH_TO_DATA_s2 = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2"
PATH_MNT = "/home/hkerino/Documents/nextoso-dataset/data/10m/mnt"
PATH_TO_DATA_label = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels"
PATH_TO_OSO_CLASSES = (
    "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
)

dict_path_to_checkpoint = {
    "LTAE": "/home/hkerino/Documents/next_oso/logs/light_training/LTAE_kfold_spatial_07-21/fold_1/Checkpoints/",
    "UTAE": "/home/hkerino/Documents/next_oso/logs/light_training/UTAE_kfold_spatial_07-16/fold_1/Checkpoints",
    "ALISE": "/home/hkerino/Documents/next_oso/logs/light_training/ALISE_kfold_spatial_07-17/fold_1/Checkpoints",
    "alpha-earth": "/home/hkerino/Documents/next_oso/logs/alpha-earth/2026-07-22_mlp_kfold/fold_1/Checkpoints",  # mlp version
    "tessera": "/home/hkerino/Documents/next_oso/logs/tessera/2026-07-22_mlp_kfold/fold_1/Checkpoints",  # mlp     version
    "malice": "/home/hkerino/Documents/next_oso/logs/malice/malice_kfold_mlp_22-07/fold_1/Checkpoints",
}


def main(config):
    model = config["model"]
    path_to_ckpt = Path(dict_path_to_checkpoint[model])
    ckpt_path = next(path_to_ckpt.glob("*.ckpt"))

    # Create directory to store predictions
    path_preds = os.path.join(args.path_preds, f"preds_{model}")
    os.makedirs(path_preds, exist_ok=True)

    if model == "alpha-earth" or model == "tessera" or model == "malice":
        print(f"{model} Model")

        datamodule = dataset_embedding.embedding_datamodule(
            data="next_oso",
            model=model,
            batch_size=1,
            fold_train=[1, 2, 3],
            fold_val=[4],
            fold_test=[5],
            return_id=True,
            mask=True,
        )
        decoder = build_decoder(
            model, args["decoder_type"], args["num_class"], dropout=0
        )
        classifier = decode_lightning_module.DecodeEmbedding.load_from_checkpoint(
            ckpt_path,
            weights_only=True,
            decoder=decoder,
            args=config,
        )

        device = next(classifier.parameters()).device
        classifier.eval()
        test_loader = datamodule.test_dataloader()

        with torch.no_grad():
            for batch in test_loader:
                embeddings, _, idx = batch
                embeddings = embeddings.to(device)
                outputs = classifier(embeddings)

                # batch_size=1
                outputs = outputs.argmax(dim=1).squeeze()

                idx = idx.item() if torch.is_tensor(idx) else idx

                torch.save(
                    outputs.cpu(),
                    os.path.join(
                        path_preds,
                        f"pred_{model}_{idx[0]}.pt",
                    ),
                )

    else:
        # Datas
        dt_test = dataset.next_oso_Dataset(
            path_s2=PATH_TO_DATA_s2,
            path_mnt=PATH_MNT,
            path_label=PATH_TO_DATA_label,
            path_oso_class=PATH_TO_OSO_CLASSES,
            norm=True,
            remove_invalid_temp=True,
            folds=[5],
            label_type="max_class",
            mask_needed=False,
            use_spatial_coords=False,
            return_id=True,
        )

        classifier = build_model(model, 1)

        it = iter(dt_test)
        classifier.eval()

        with torch.no_grad():
            while True:
                try:
                    inputs, _, idx = next(it)
                    outputs = classifier(inputs)
                    outputs = outputs.argmax(dim=1).squeeze()
                    torch.save(
                        outputs,
                        os.path.join(
                            path_preds,
                            f"pred_{model}_{idx}.pt",
                        ),
                    )

                except StopIteration:
                    break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_train", type=str, default="configs/train_local.yaml")
    parser.add_argument(
        "--config_model", type=str, required=True, help="Ex: configs/models/LTAE.yaml"
    )
    parser.add_argument(
        "--path_preds",
        type=str,
        required=True,
        help="/home/hkerino/Documents/Resultats_nextoso/preds_folder_5",
    )

    args = parser.parse_args()
    config = load_config(args.config_train, args.config_model)
    pprint.pprint(config)

    main(config)
