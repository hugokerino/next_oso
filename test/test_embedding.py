import torch

from embedding import dataset_embedding, decode_lightning_module
from src.utils_model import MLP_Decoder, CNN_Decoder


def test_embedding():
    model = "malice"  # alpha-eath, tessera, malice
    datamodule = dataset_embedding.embedding_datamodule(
        data="next_oso",
        model=model,
        path_embeddings="/home/hkerino/Documents/model_data/MALICE",  # "/home/hkerino/Documents/nextoso-dataset",
        path_labels="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
        path_mapping_folder="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
        fold_train=[1],
        fold_val=[2],
        fold_test=[3],
        batch_size=2,
        num_workers=1,
        preload=False,
        label_type="max_class",
        mask=True,
    )

    if model == "alpha-earth":
        layers = [64, 32, 25]
    elif model == "tessera":
        layers = [128, 64, 32, 25]
    elif model == "malice":
        layers = [640, 320, 64, 25]
    else:
        print(f"{model} not recognized")

    config = {
        "data": "next_oso",
        "lr": 1e-3,
        "max_epochs": 2,
        "path_name_class": "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv",
        "label_type": "max_class",
        "path_save_excel": None,
    }

    classifier_MLP = decode_lightning_module.DecodeEmbedding(
        decoder=MLP_Decoder(layers=layers, dropout=0.1), config=config
    )
    classifier_CNN = decode_lightning_module.DecodeEmbedding(
        decoder=CNN_Decoder(layers=layers, dropout=0.1), config=config
    )

    # Training step
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    loss_MLP = classifier_MLP.training_step(batch, batch_idx=0)
    loss_CNN = classifier_CNN.training_step(batch, batch_idx=0)
    assert isinstance(loss_MLP, torch.Tensor)
    assert isinstance(loss_CNN, torch.Tensor)

    # Validation step
    val_batch = next(iter(datamodule.val_dataloader()))
    classifier_MLP.validation_step(val_batch, batch_idx=0)
    classifier_CNN.validation_step(val_batch, batch_idx=0)

    # Test step
    test_batch = next(iter(datamodule.test_dataloader()))
    classifier_MLP.test_step(test_batch)
    classifier_CNN.test_step(test_batch)


def test_embedding_pastis():
    config = {
        "data": "pastis",
        "lr": 1e-3,
        "max_epochs": 2,
        "path_name_class": "/home/hkerino/Documents/PASTIS/label_names_reduced.json",
        "label_type": None,
        "path_save_excel": None,
    }

    model = "alpha-earth"  # alpha-eath, tessera, malice
    datamodule = dataset_embedding.embedding_datamodule(
        data="pastis",
        model=model,
        path_embeddings="/home/hkerino/Documents/PASTIS",  # "/home/hkerino/Documents/nextoso-dataset",
        path_labels="/home/hkerino/Documents/PASTIS/ANNOTATIONS",
        path_mapping_folder="/home/hkerino/Documents/PASTIS",
        fold_train=[1],
        fold_val=[2],
        fold_test=[3],
        batch_size=2,
        num_workers=1,
        preload=False,
        label_type=None,
        mask=True,
    )

    if model == "alpha-earth":
        layers = [64, 32, 25]
    elif model == "tessera":
        layers = [128, 64, 32, 25]
    elif model == "malice":
        layers = [640, 320, 64, 25]
    else:
        print(f"{model} not recognized")

    classifier_MLP = decode_lightning_module.DecodeEmbedding(
        decoder=MLP_Decoder(layers=layers, dropout=0.1), config=config
    )
    classifier_CNN = decode_lightning_module.DecodeEmbedding(
        decoder=CNN_Decoder(layers=layers, dropout=0.1), config=config
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Training step
    loss_MLP = classifier_MLP.training_step(batch, batch_idx=0)
    loss_CNN = classifier_CNN.training_step(batch, batch_idx=0)
    assert isinstance(loss_MLP, torch.Tensor)
    assert isinstance(loss_CNN, torch.Tensor)

    # Validation step
    val_batch = next(iter(datamodule.val_dataloader()))
    classifier_MLP.validation_step(val_batch, batch_idx=0)
    classifier_CNN.validation_step(val_batch, batch_idx=0)

    # Test step
    test_batch = next(iter(datamodule.test_dataloader()))
    classifier_MLP.test_step(test_batch)
    classifier_CNN.test_step(test_batch)
