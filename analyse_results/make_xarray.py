import torch
import yaml
from argparse import ArgumentParser
import xarray as xr
from collections import defaultdict
import rasterio
import numpy as np
import pprint

from src import lightning_module, LTAE, UTAE, utils_model
from Alise import ALISE, sse, temporal_proj, transformer, utils_ALISE
from embedding import dataset_embedding, decode_lightning_module
from mains.main_embedding import MLP_Decoder
from data_functions import dataset

BASE_PATH = (
    "/work/THEIA/CES/oso/hugok/next_oso"
    if torch.cuda.is_available()
    else "/home/hkerino/Documents/next_oso"
)


def select_model(name_model, batch_size, path_to_checkpoint=None):
    criterion_ce = torch.nn.CrossEntropyLoss()

    if name_model == "LTAE":
        with open(BASE_PATH + "/configs/models/LTAE.yaml", "r") as f:
            config_ltae = yaml.safe_load(f)

        encoder = LTAE.LTAE2d(**config_ltae["encoder"])
        decoder = utils_model.MLPDecoder(**config_ltae["decoder"])
        if path_to_checkpoint is None:
            checkpoint = "/home/hkerino/Documents/next_oso/logs/light_training/LTAE_2026-05-28/Checkpoints/epoch=124-step=135250.ckpt"
        else:
            checkpoint = path_to_checkpoint

    if name_model == "UTAE":
        with open(BASE_PATH + "/configs/models/UTAE.yaml", "r") as f:
            config_utae = yaml.safe_load(f)

        encoder = UTAE.UTAE(**config_utae["encoder"])
        decoder = utils_model.MLPDecoder(**config_utae["decoder"])
        if path_to_checkpoint is None:
            checkpoint = "/home/hkerino/Documents/next_oso/logs/light_training/UTAE_2026-04-09_1/Checkpoints/epoch=51-step=14092.ckpt"
        else:
            checkpoint = path_to_checkpoint

    if name_model == "ALISE":
        with open(BASE_PATH + "/configs/models/ALISE.yaml", "r") as f:
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
        if path_to_checkpoint is None:
            checkpoint = "/home/hkerino/Documents/next_oso/logs/light_training/ALISE_2026-05-27/Checkpoints/epoch=36-step=80031.ckpt"
        else:
            checkpoint = path_to_checkpoint

    return lightning_module.classifier.load_from_checkpoint(
        checkpoint_path=checkpoint,
        weights_only=True,
        encoder=encoder,
        decoder=decoder,
        loss_fn=criterion_ce,
        batch_size=batch_size,
    )


def build_MLP_embedding(model, path_to_checkpoint=None):
    class Args:
        def __init__(self, time_feature=False):
            self.num_class = 25
            self.lr = 1e-3
            self.max_epochs = 2
            self.path_name_class = "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
            self.label_type = "max_class"
            self.time_feature = time_feature

    if model == "alpha-earth":
        if path_to_checkpoint is None:
            checkpoint = "/home/hkerino/Documents/next_oso/logs/alpha-earth/2026-06-18/Checkpoints/epoch=398-step=13167.ckpt"
        else:
            checkpoint = path_to_checkpoint
        mlp = [64, 64, 32, 25]
        time_feature = False

    elif model == "tessera":
        if path_to_checkpoint is None:
            checkpoint = "/home/hkerino/Documents/next_oso/logs/tessera/2026-06-18/Checkpoints/epoch=182-step=6222.ckpt"
        else:
            checkpoint = path_to_checkpoint
        mlp = [128, 64, 32, 25]
        time_feature = False

    elif model == "malice":
        if path_to_checkpoint is None:
            checkpoint = "/home/hkerino/Documents/next_oso/logs/malice/2026-07-07/Checkpoints/epoch=106-step=28997.ckpt"
        else:
            checkpoint = path_to_checkpoint
        mlp = [640, 320, 64, 25]
        time_feature = True

    classifier = decode_lightning_module.DecodeEmbedding.load_from_checkpoint(
        checkpoint_path=checkpoint,
        decoder=MLP_Decoder(mlp=mlp, dropout=0.1),
        args=Args(time_feature=time_feature),
    )
    return classifier


def sanity_check(models, config):
    # To prevent error when loading models, we check if the model can be loaded successfully
    for model in models:
        try:
            if model == "alpha-earth" or model == "tessera" or model == "malice":
                _ = build_MLP_embedding(
                    model, path_to_checkpoint=config["BASE_PATH"] + config[model]
                )
            else:
                _ = select_model(
                    model,
                    batch_size=1,
                    path_to_checkpoint=config["BASE_PATH"] + config[model],
                )
        except Exception as e:
            print(f"Error loading model {model}: {e}")
            raise


def main(config):
    # Storage for netcdf creation
    store_pred = defaultdict(dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    models = ["alpha-earth", "tessera", "malice", "LTAE", "UTAE", "ALISE"]

    sanity_check(models, config)

    for model in models:
        # Create directory to store predictions
        print(f"\nProcessing {model} prediction")

        if model == "alpha-earth" or model == "tessera" or model == "malice":
            datamodule = dataset_embedding.embedding_datamodule(
                path_embeddings=config["PATH_TO_EMBEDDINGS"] + f"/{model}",
                path_labels=config["PATH_TO_LABELS"],
                path_mapping_folder=config["PATH_TO_MAPPING_FOLDER"],
                model=model,
                batch_size=1,
                fold_train=[1, 2, 3],
                fold_val=[4],
                fold_test=[5],
                return_id=True,
                mask=True,
            )
            classifier = build_MLP_embedding(
                model, path_to_checkpoint=config["BASE_PATH"] + config[model]
            )
            classifier.eval()
            test_loader = datamodule.test_dataloader()

            with torch.no_grad():
                for batch in test_loader:
                    embeddings, _, idx = batch
                    # Move embeddings to GPU
                    embeddings = embeddings.to(device).float()
                    classifier = classifier.to(device)
                    probas = torch.softmax(
                        classifier(embeddings).squeeze(), dim=0
                    )  # HWC
                    outputs = probas.argmax(dim=0)  # HW
                    probas = torch.round(probas * 100)

                    store_pred[int(idx[0])][model] = {
                        "probas": probas.cpu().numpy().astype(np.int8),
                        "preds": outputs.cpu().numpy().astype(np.int8),
                    }
        else:
            # Datas
            dt_test = dataset.next_oso_Dataset(
                path_s2=config["PATH_TO_s2"],
                path_mnt=config["PATH_TO_mnt"],
                path_label=config["PATH_TO_LABELS"],
                path_oso_class=config["PATH_TO_oso_classes"],
                norm=True,
                remove_invalid_temp=True,
                folds=[5],
                label_type="max_class",
                mask_needed=False,
                return_id=True,
            )

            classifier = select_model(
                model,
                batch_size,
                path_to_checkpoint=config["BASE_PATH"] + config[model],
            )

            it = iter(dt_test)
            classifier.eval()

            with torch.no_grad():
                while True:
                    try:
                        inputs, _, idx = next(it)
                        # Move inputs to GPU
                        inputs = tuple(
                            x.to(device) if isinstance(x, torch.Tensor) else x
                            for x in inputs
                        )
                        classifier = classifier.to(device)
                        probas = torch.softmax(classifier(inputs).squeeze(), dim=0)
                        outputs = probas.argmax(dim=0)
                        probas = torch.round(probas * 100)

                        store_pred[int(idx)][model] = {
                            "probas": probas.cpu().numpy().astype(np.int8),
                            "preds": outputs.cpu().numpy().astype(np.int8),
                        }
                    except StopIteration:
                        break

    print("\nSave netcdf files")
    for index in sorted(store_pred.keys()):
        ds = xr.open_dataset(f"{config['PATH_TO_LABELS']}/label_{index:04d}.nc")
        ny = len(ds.y)
        nx = len(ds.x)
        n_classes = 25

        for model in models:
            if model in store_pred[index]:
                preds = store_pred[index][model]["preds"]
                probas = store_pred[index][model]["probas"]

            else:
                preds = np.full((ny, nx), 0, dtype=np.int8)

                probas = np.full((n_classes, ny, nx), 0, dtype=np.int8)

            ds[f"pred_{model}"] = xr.DataArray(
                preds,
                dims=("y", "x"),
            )
            ds[f"proba_{model}"] = xr.DataArray(
                probas,
                dims=("class", "y", "x"),
            )

        with rasterio.open(f"{config['PATH_TO_RGB_IMG']}/s2_{index:04d}.tif") as src:
            s2 = src.read()
            ds["sentinel2_img"] = xr.DataArray(
                s2,
                dims=("band", "y", "x"),
            )
        path_to_save = f"{config['PATH_TO_SAVE_netcdf']}/{index:04d}.nc"

        for var_name in ds.keys():
            ds[var_name].attrs["grid_mapping"] = "crs"

        ds.to_netcdf(path_to_save, mode="w")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_models", type=str, default="configs/models/path_to_models.yaml"
    )
    args = parser.parse_args()

    with open(args.config_models, "r") as f:
        config = yaml.safe_load(f)

    pprint.pprint(config)

    main(config)
