import torch
import os
import yaml
from argparse import ArgumentParser

from src import lightning_module, LTAE, UTAE, utils_model
from Alise import ALISE, sse, temporal_proj, transformer, utils_ALISE
from embedding import dataset_embedding, decode_lightning_module
from mains.main_embedding import MLP_Decoder
from data_functions import dataset

PATH_TO_DATA_s2 = "/home/hkerino/Documents/nextoso-dataset/data/10m/s2"
PATH_MNT = "/home/hkerino/Documents/nextoso-dataset/data/10m/mnt"
PATH_TO_DATA_label = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels"
PATH_TO_OSO_CLASSES = (
    "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
)


def select_model(name_model, batch_size):
    criterion_ce = torch.nn.CrossEntropyLoss()

    if name_model == "LTAE":
        with open(
            "/home/hkerino/Documents/next_oso/configs/models/LTAE.yaml", "r"
        ) as f:
            config_ltae = yaml.safe_load(f)

        encoder = LTAE.LTAE2d(**config_ltae["encoder"])
        decoder = utils_model.MLPDecoder(**config_ltae["decoder"])
        checkpoint = "/home/hkerino/Documents/next_oso/logs/light_training/LTAE_2026-05-28/Checkpoints/epoch=124-step=135250.ckpt"

    if name_model == "UTAE":
        with open(
            "/home/hkerino/Documents/next_oso/configs/models/UTAE.yaml", "r"
        ) as f:
            config_utae = yaml.safe_load(f)

        encoder = UTAE.UTAE(**config_utae["encoder"])
        decoder = utils_model.MLPDecoder(**config_utae["decoder"])
        checkpoint = "/home/hkerino/Documents/next_oso/logs/light_training/UTAE_2026-04-09_1/Checkpoints/epoch=51-step=14092.ckpt"

    if name_model == "ALISE":
        with open(
            "/home/hkerino/Documents/next_oso/configs/models/ALISE.yaml", "r"
        ) as f:
            config_alise = yaml.safe_load(f)

        enc_cfg = config_alise["encoder"]
        emb_dim = enc_cfg["embedding_dim"]
        encoder = ALISE.ALISE(
            patch_embedding=sse.PatchEmbedding(
                sse=sse.Unet(planes=emb_dim, **enc_cfg["sse"]),
                tpe=sse.PositionalEncoder(d=emb_dim),
            ),
            temporal_encoder=transformer.Transformer(**enc_cfg["transformer"]),
            temporal_projector=temporal_proj.LearnedQMultiHeadAttention(
                d_in=emb_dim, **enc_cfg["temporal_projector"]
            ),
        )
        decoder = utils_ALISE.MLP_Simple_Decoder(**config_alise["decoder"])
        checkpoint = "/home/hkerino/Documents/next_oso/logs/light_training/ALISE_2026-05-27/Checkpoints/epoch=36-step=80031.ckpt"

    return lightning_module.classifier.load_from_checkpoint(
        checkpoint_path=checkpoint,
        weights_only=True,
        encoder=encoder,
        decoder=decoder,
        loss_fn=criterion_ce,
        batch_size=batch_size,
    )


def build_MLP_embedding(model):
    class Args:
        def __init__(self, time_feature=False):
            self.num_class = 25
            self.lr = 1e-3
            self.max_epochs = 1
            self.path_name_class = "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
            self.label_type = "max_class"
            self.time_feature = time_feature

    if model == "alpha-earth":
        checkpoint = "/home/hkerino/Documents/next_oso/logs/alpha-earth/2026-06-18/Checkpoints/epoch=398-step=13167.ckpt"
        mlp = [64, 64, 32, 25]
        time_feature = False

    elif model == "tessera":
        checkpoint = "/home/hkerino/Documents/next_oso/logs/tessera/2026-06-18/Checkpoints/epoch=182-step=6222.ckpt"
        mlp = [128, 64, 32, 25]
        time_feature = False

    elif model == "malice":
        checkpoint = "/work/THEIA/CES/oso/hugok/next_oso/logs/malice/malice/2026-07-07/Checkpoints/epoch=106-step=28997.ckpt"  # "/home/hkerino/Documents/next_oso/logs/malice/2026-07-07/Checkpoints/epoch=106-step=28997.ckpt"
        mlp = [640, 320, 64, 25]
        time_feature = True
    else:
        raise ValueError(f"Model {model} not recognized")

    classifier = decode_lightning_module.DecodeEmbedding.load_from_checkpoint(
        checkpoint_path=checkpoint,
        decoder=MLP_Decoder(mlp=mlp, dropout=0.1),
        args=Args(time_feature=time_feature),
    )
    return classifier


def main(args):
    model, batch_size = args.model, args.batch_size

    # Create directory to store predictions
    path_preds = os.path.join(args.path_preds, f"preds_{model}")
    os.makedirs(path_preds, exist_ok=True)

    if model == "alpha-earth" or model == "tessera" or model == "malice":
        print(f"{model} Model")

        datamodule = dataset_embedding.embedding_datamodule(
            path_embeddings=args.path_embedding,
            path_labels=args.path_label,
            path_mapping_folder=args.path_mapping_folder,
            model=model,
            batch_size=1,
            fold_train=[1, 2, 3],
            fold_val=[4],
            fold_test=[5],
            return_id=True,
            mask=True,
        )
        classifier = build_MLP_embedding(model)
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

        classifier = select_model(model, batch_size)

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
    parser.add_argument("--model", type=str, required=True, help="Ex: LTAE")
    parser.add_argument(
        "--path_embedding",
        type=str,
        required=False,
        default="/home/hkerino/Documents/nextoso-dataset",
    )
    parser.add_argument(
        "--path_label",
        type=str,
        required=False,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    )
    parser.add_argument(
        "--path_mapping_folder",
        type=str,
        required=False,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    )
    parser.add_argument(
        "--path_preds",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/preds",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
