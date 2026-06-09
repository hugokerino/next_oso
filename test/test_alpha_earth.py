import pytest
import sys
import torch
import random

sys.path.append("/home/hkerino/Documents/next_oso/")

from alpha_earth import dataset_alpha, decode_lightning_module
from mains.main_alpha_earth import MLP_Decoder


class Args:
    def __init__(self):
        self.num_class = 25
        self.lr = 1e-3
        self.max_epochs = 2
        self.path_name_class = (
            "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
        )
        self.label_type = "probas"


def test_alpha_earth():
    args = Args()

    datamodule = dataset_alpha.alpha_earth_datamodule(
        path_embeddings="/home/hkerino/Documents/nextoso-dataset/alpha-earth_data",
        path_labels="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
        path_mapping_folder="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
        fold_train=[1],
        fold_val=[2],
        fold_test=[3],
        batch_size=8,
        num_workers=1,
        preload=False,
        label_type="probas",
    )

    classifier = decode_lightning_module.DecodeEmbedding(
        decoder=MLP_Decoder(mlp=[64, 32, 25], dropout=0.1), args=args
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    loss = classifier.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)

    val_batch = next(iter(datamodule.val_dataloader()))
    classifier.validation_step(val_batch, batch_idx=0)

    test_batch = next(iter(datamodule.test_dataloader()))
    classifier.test_step(test_batch)
