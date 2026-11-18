import yaml
import pytest

import torch
from embedding.dataset_embedding import EmbeddingDataModule, Embedding_Dataset
from embedding.dataset_embedding_PCA import EmbeddingDatamodulePCA
from embedding.decode_lightning_module import DecodeEmbedding

from mains.main_utils import build_decoder

path_config = "./configs/embeddings/config_local.yaml"
with open(path_config, "r") as f:
    train_cfg = yaml.safe_load(f)
config = {**train_cfg}


def test_embedding(config=config):
    model = "alpha-earth"  # alpha-earth, tessera, malice
    datamodule = EmbeddingDataModule(
        dataset="next_oso",
        path_data="/home/hkerino/Documents/nextoso-dataset/data/10m",
        model=model,
        fold_train=[1],
        fold_val=[2],
        fold_test=[3],
        batch_size=config["batch_size"],
        num_workers=1,
        preload=False,
        label_type="max_class",
        mask=True,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    decoder_mlp = build_decoder(
        model=model,
        decoder_type="mlp",
        num_class=config["num_class"],
        dropout=config["dropout"],
    )
    classifier_MLP = DecodeEmbedding(
        decoder=decoder_mlp,
        loss_fn=criterion,
        batch_size=config["batch_size"],
        num_class=config["num_class"],
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        optimizer_scheduler=config["optimizer_scheduler"],
        names_class=datamodule.data_train.name_labels,
    )

    decoder_cnn = build_decoder(
        model=model,
        decoder_type="cnn",
        num_class=config["num_class"],
        dropout=config["dropout"],
    )
    classifier_CNN = DecodeEmbedding(
        decoder=decoder_cnn,
        loss_fn=criterion,
        batch_size=config["batch_size"],
        num_class=config["num_class"],
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        optimizer_scheduler=config["optimizer_scheduler"],
        names_class=datamodule.data_train.name_labels,
    )

    # Training step
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    loss_MLP = classifier_MLP.training_step(batch)
    loss_CNN = classifier_CNN.training_step(batch)
    assert isinstance(loss_MLP, torch.Tensor)
    assert isinstance(loss_CNN, torch.Tensor)

    # Validation step
    val_batch = next(iter(datamodule.val_dataloader()))
    classifier_MLP.validation_step(val_batch)
    classifier_CNN.validation_step(val_batch)

    # Test step
    test_batch = next(iter(datamodule.test_dataloader()))
    classifier_MLP.test_step(test_batch)
    classifier_CNN.test_step(test_batch)


def test_embedding_pastis(config=config):
    model = "alpha-earth"  # alpha-earth, tessera, malice
    num_class_pastis = 19
    datamodule = EmbeddingDataModule(
        dataset="pastis",
        model=model,
        path_data="/home/hkerino/Documents/PASTIS",
        fold_train=[1],
        fold_val=[2],
        fold_test=[3],
        batch_size=config["batch_size"],
        num_workers=1,
        preload=False,
        label_type=None,
        mask=True,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    decoder_mlp = build_decoder(
        model=model,
        decoder_type="mlp",
        num_class=num_class_pastis,
        dropout=config["dropout"],
    )
    classifier_MLP = DecodeEmbedding(
        decoder=decoder_mlp,
        loss_fn=criterion,
        batch_size=config["batch_size"],
        num_class=num_class_pastis,
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        optimizer_scheduler=config["optimizer_scheduler"],
        names_class=datamodule.data_train.name_labels,
    )

    decoder_cnn = build_decoder(
        model=model,
        decoder_type="cnn",
        num_class=num_class_pastis,
        dropout=config["dropout"],
    )
    classifier_CNN = DecodeEmbedding(
        decoder=decoder_cnn,
        loss_fn=criterion,
        batch_size=config["batch_size"],
        num_class=num_class_pastis,
        lr=config["lr"],
        max_epochs=config["max_epochs"],
        optimizer_scheduler=config["optimizer_scheduler"],
        names_class=datamodule.data_train.name_labels,
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Training step
    loss_MLP = classifier_MLP.training_step(batch)
    loss_CNN = classifier_CNN.training_step(batch)
    assert isinstance(loss_MLP, torch.Tensor)
    assert isinstance(loss_CNN, torch.Tensor)

    # Validation step
    val_batch = next(iter(datamodule.val_dataloader()))
    classifier_MLP.validation_step(val_batch)
    classifier_CNN.validation_step(val_batch)

    # Test step
    test_batch = next(iter(datamodule.test_dataloader()))
    classifier_MLP.test_step(test_batch)
    classifier_CNN.test_step(test_batch)


def test_raise_errors():
    with pytest.raises(
        ValueError,
        match="Dataset error.",
    ):
        Embedding_Dataset(dataset="Not_implemented")

    with pytest.raises(
        ValueError,
        match="Model error.",
    ):
        Embedding_Dataset(model="Not_implemented")


##PCA version
def test_embedding_pca(config=config):
    model = "tessera"  # alpha-earth, tessera, malice
    datamodule = EmbeddingDatamodulePCA(
        n_component=32,
        percentage_sample=0.01,
        dataset="next_oso",
        path_data="/home/hkerino/Documents/nextoso-dataset/data/10m",
        model=model,
        fold_train=[1],
        fold_val=[2],
        fold_test=[3],
        batch_size=config["batch_size"],
        num_workers=1,
        preload=False,
        label_type="max_class",
        mask=True,
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    inputs, labels = batch
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    with pytest.raises(
        ValueError,
        match="Number of component to keep cannot be larger than initial one.",
    ):
        EmbeddingDatamodulePCA(
            n_component=10000,  # Should detect error here
            percentage_sample=0.01,
            dataset="next_oso",
            path_data="/home/hkerino/Documents/nextoso-dataset/data/10m",
            model=model,
            fold_train=[1],
            fold_val=[2],
            fold_test=[3],
            batch_size=config["batch_size"],
            num_workers=1,
            preload=False,
            label_type="max_class",
            mask=True,
        )

    with pytest.raises(
        ValueError,
        match="Percentage should be between 0.0 and 1.0.",
    ):
        EmbeddingDatamodulePCA(
            n_component=32,
            percentage_sample=-10000,  # Should detect error here
            dataset="next_oso",
            path_data="/home/hkerino/Documents/nextoso-dataset/data/10m",
            model=model,
            fold_train=[1],
            fold_val=[2],
            fold_test=[3],
            batch_size=config["batch_size"],
            num_workers=1,
            preload=False,
            label_type="max_class",
            mask=True,
        )
