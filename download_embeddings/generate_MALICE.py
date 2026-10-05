import os

import onnxruntime
import torch
from einops import rearrange
from argparse import ArgumentParser, Namespace
import numpy as np

from data_functions.transfrom import load_transform_one_mod
from data_functions import dataset
from pastis.pastis_dataset import PASTIS_Dataset
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_series_malise_onnx(
    args: Namespace,
    sat: str = "S2",
):
    """Encode PASTIS SITS S2 with MALICE with exported onnx model"""

    # MALICE MODEL
    sess_opt = onnxruntime.SessionOptions()
    sess_opt.intra_op_num_threads = 16
    transform = load_transform_one_mod(args.path_csv_norm, mod=sat.lower()).transform

    ort_session = onnxruntime.InferenceSession(
        args.path_malice_model,
        sess_opt,
        providers=["CPUExecutionProvider"],
    )

    if args.data == "next_oso":
        # DATA
        d = dataset.next_oso_Dataset(
            path_data=args.path_data,
            folds=[args.fold],
            norm=False,
            return_id=True,
        )
        nbr_sample = d.__len__()

        for i in range(nbr_sample):
            (s2, dates, _, _), _, id = d.__getitem__(i)

            save_path = f"{args.path_saved_embeddings}/MALICE_{id}.pt"
            if os.path.exists(save_path):
                continue

            # Prepare patch
            ref_norm = rearrange(transform(s2), "c t h w -> 1 t c h w")

            input = {
                "sits": ref_norm.cpu().numpy(),
                "tpe": dates.numpy().astype(np.float32)[None, :],
                "padd_mask": torch.zeros(1, ref_norm.shape[1]).bool().numpy(),
            }

            ort_out = ort_session.run(None, input)[0]  # (1, 10, 64, 64, 64)

            # Save the embedding
            torch.save(torch.Tensor(ort_out[0]), save_path)
            print(f"Sample {id} processed")
            sys.stdout.flush()

    if args.data == "pastis":
        # DATA
        p = PASTIS_Dataset(folder=args.path_pastis, norm=False, return_id=True)
        nbr_sample = p.__len__()

        for i in range(nbr_sample):
            (s2, dates_), _, id = p.__getitem__(i)

            save_path = f"{args.path_saved_embeddings}/MALICE_{id}.pt"
            if os.path.exists(save_path):
                continue

            ref_norm = (
                transform(s2["S2"].permute(1, 0, 2, 3).contiguous())
                .permute(1, 0, 2, 3)
                .contiguous()
                .unsqueeze(dim=0)
            )
            dates = dates_["S2"]

            input = {
                "sits": ref_norm.cpu().numpy(),
                "tpe": dates.numpy().astype(np.float32)[None, :],
                "padd_mask": torch.zeros(1, ref_norm.shape[1]).bool().numpy(),
            }

            ort_out = ort_session.run(None, input)[0]  # (1, 10, 64, 64, 64)

            # Save the embedding
            torch.save(torch.Tensor(ort_out[0]), save_path)
            print(f"Sample {id} processed")
            sys.stdout.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        choices=["next_oso", "pastis"],
        required=True,
    )

    parser.add_argument(
        "--path_data",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/",
    )

    parser.add_argument(
        "--path_saved_embeddings",
        type=str,
        default="/home/hkerino/Documents/PASTIS/MALICE_DATA",
        # default="/home/hkerino/Documents/model_data/MALICE/embeddings",
    )
    parser.add_argument(
        "--path_csv_norm",
        type=str,
        default="/home/hkerino/Documents/model_data/MALICE/norm_MALICE_s2.csv",
    )
    parser.add_argument(
        "--path_malice_model",
        type=str,
        default="/home/hkerino/Documents/model_data/MALICE/malice_s2.onnx",
    )
    parser.add_argument("--fold", type=int, default=1)

    args = parser.parse_args()
    encode_series_malise_onnx(
        sat="S2",
        args=args,
    )
