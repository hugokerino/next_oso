import torch

from embedding import dataset_embedding, decode_lightning_module
from mains.main_embedding import MLP_Decoder


class Args:
    def __init__(
        self,
        num_class: int = 25,
        lr: float = 1e-3,
        max_epochs: int = 2,
        path_name_class: str = (
            "/home/hkerino/Documents/nextoso-dataset/data/auxiliary/OSO/oso_classes.csv"
        ),
        label_type: str = "max_class",
        time_feature: bool = True,
        path_save_excel=None,
    ):
        self.num_class = num_class
        self.lr = lr
        self.max_epochs = max_epochs
        self.path_name_class = path_name_class
        self.label_type = label_type
        self.time_feature = time_feature
        self.path_save_excel = path_save_excel


def test_embedding():
    args = Args()
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
        mlp = [64, 32, 25]
    elif model == "tessera":
        mlp = [128, 64, 32, 25]
    elif model == "malice":
        mlp = [640, 320, 64, 25]
    else:
        print(f"{model} not recognized")

    classifier = decode_lightning_module.DecodeEmbedding(
        decoder=MLP_Decoder(mlp=mlp, dropout=0.1), args=args
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    loss = classifier.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)

    val_batch = next(iter(datamodule.val_dataloader()))
    classifier.validation_step(val_batch, batch_idx=0)

    test_batch = next(iter(datamodule.test_dataloader()))
    classifier.test_step(test_batch)


def test_embedding_pastis():
    args = Args(num_class=19, label_type=None, path_save_excel=None, time_feature=False)
    model = "alpha-eath"  # alpha-eath, tessera, malice
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
        mlp = [64, 32, 25]
    elif model == "tessera":
        mlp = [128, 64, 32, 25]
    elif model == "malice":
        mlp = [640, 320, 64, 25]
    else:
        print(f"{model} not recognized")

    classifier = decode_lightning_module.DecodeEmbedding(
        decoder=MLP_Decoder(mlp=mlp, dropout=0.1), args=args
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    loss = classifier.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)

    val_batch = next(iter(datamodule.val_dataloader()))
    classifier.validation_step(val_batch, batch_idx=0)

    test_batch = next(iter(datamodule.test_dataloader()))
    classifier.test_step(test_batch)
