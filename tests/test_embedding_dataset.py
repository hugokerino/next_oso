import pytest
import yaml
import torch
from embedding.dataset_embedding_PCA import EmbeddingDatamodulePCA
from embedding.dataset_embedding import Embedding_Dataset

path_dataset_next_oso = "/home/hkerino/Documents/nextoso-dataset/data/10m"  # to change
path_dataset_pastis = "/home/hkerino/Documents/PASTIS"  # to change


def test_dataset_default():
    models = ["alpha-earth", "tessera", "malice"]
    datasets_path = {"next_oso": path_dataset_next_oso, "pastis": path_dataset_pastis}

    for dataset, path in datasets_path.items():
        for model in models:
            d = Embedding_Dataset(path_data=path, model=model, dataset=dataset)
            sample = d.__getitem__(0)

            assert isinstance(sample, tuple)
            assert len(sample) == 2  # (datas , label)
            assert isinstance(sample[0], torch.Tensor)
            assert isinstance(sample[1], torch.Tensor)


def test_dataset_choose_folds():
    d1 = Embedding_Dataset(path_data=path_dataset_next_oso, folds=[1])
    d2 = Embedding_Dataset(path_data=path_dataset_next_oso, folds=[1, 5])
    d3 = Embedding_Dataset(path_data=path_dataset_next_oso, folds=[1, 2, 3, 4, 5])
    d4 = Embedding_Dataset(path_data=path_dataset_next_oso, folds=None)

    assert d1.__len__() < d2.__len__()
    assert d3.__len__() == d4.__len__()


def test_raise_errors():
    with pytest.raises(
        ValueError,
        match="Dataset error.",
    ):
        Embedding_Dataset(path_data=path_dataset_next_oso, dataset="Not_implemented")

    with pytest.raises(
        ValueError,
        match="Model error.",
    ):
        Embedding_Dataset(path_data=path_dataset_next_oso, model="Not_implemented")


path_config = "./configs/embeddings/config_local.yaml"
with open(path_config, "r") as f:
    train_cfg = yaml.safe_load(f)
config = {**train_cfg}


##PCA version
def test_embedding_pca(config=config):
    model = "tessera"  # alpha-earth, tessera, malice
    datamodule = EmbeddingDatamodulePCA(
        n_component=32,
        percentage_sample=0.01,
        dataset="next_oso",
        path_data=path_dataset_next_oso,
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
            path_data=path_dataset_next_oso,
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
            path_data=path_dataset_next_oso,
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
