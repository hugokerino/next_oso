import torch
from embedding import dataset_embedding, decode_lightning_module
from supervised_models.utils_model import MLP_Decoder, CNN_Decoder


def test_embedding():
    model = "alpha-earth"  # alpha-earth, tessera, malice
    datamodule = dataset_embedding.embedding_datamodule(
        dataset="next_oso",
        path_data="/home/hkerino/Documents/nextoso-dataset/data/10m",
        model=model,
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
        "dataset": "next_oso",
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


def test_embedding_pastis():
    config = {
        "dataset": "pastis",
        "lr": 1e-3,
        "max_epochs": 2,
        "path_name_class": "/home/hkerino/Documents/PASTIS/label_names_reduced.json",
        "label_type": None,
        "path_save_excel": None,
    }

    model = "alpha-earth"  # alpha-earth, tessera, malice
    datamodule = dataset_embedding.embedding_datamodule(
        dataset="pastis",
        model=model,
        path_data="/home/hkerino/Documents/PASTIS",
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
