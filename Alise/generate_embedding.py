import torch
import onnxruntime as ort
import sys
import os
from argparse import ArgumentParser

from data_functions import dataset
from Alise import utils_ALISE


def generate_embedding(args):

    # Data loading
    d = dataset.next_oso_Dataset(
        path_s2=args.path_s2,
        path_mnt=args.path_mnt,
        path_label=args.path_label,
        path_oso_class=args.path_oso_classes,
        folds=[args.fold],
        norm=False,
        norm_pretrained_alise=True,
        mask_needed=False,
        use_spatial_coords=False,
        return_id=True,
    )

    nbr_sample = d.__len__()

    # Load onnx model
    session = ort.InferenceSession(args.path_model_onnx)
    alise = utils_ALISE.ONNXEncoderWrapper(session)

    os.makedirs(args.path_saved_embeddings, exist_ok=True)

    for i in range(nbr_sample):
        (s2, dates, _, _), _, id = d.__getitem__(i)

        save_path = f"{args.path_saved_embeddings}/emb_{id}.pt"
        if os.path.exists(save_path):
            continue

        embedding = alise(s2.unsqueeze(0), dates.unsqueeze(0)).squeeze(0)  # (t,c,h,w)

        # Save the embedding
        torch.save(embedding.cpu(), save_path)
        print(f"Sample {id} processed")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path_s2",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
    )
    parser.add_argument(
        "--path_mnt",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/mnt",
    )
    parser.add_argument(
        "--path_label",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
    )
    parser.add_argument(
        "--path_oso_classes",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
    )
    parser.add_argument(
        "--path_saved_embeddings",
        type=str,
        default="/home/hkerino/Documents/model_data/embeddings",
    )

    parser.add_argument(
        "--path_model_onnx",
        type=str,
        default="/home/hkerino/Documents/model_data/alise_weights.onnx",
    )
    parser.add_argument("--fold", type=int, required=True)

    args = parser.parse_args()
    generate_embedding(args)
