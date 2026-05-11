import pytest
import sys

sys.path.append("/home/hkerino/Documents/next_oso/")

from Alise import decode_embedding
import torch


@pytest.mark.parametrize("fold", [[1], [5], None, [1, 2, 3], [1, 2, 3, 4, 5]])
def test_data_embedding(fold):

    dataset = decode_embedding.alise_embedding_dataset(
        path_embeddings="/home/hkerino/Documents/model_data/embeddings",
        path_labels="/home/hkerino/Documents/nextoso-dataset/data/10m/labels",
        path_mapping_folder="/home/hkerino/Documents/nextoso-dataset/data/10m/s2",
        fold=fold,
        label_type="max_class",
    )

    emb, label = dataset.__getitem__(0)

    assert dataset.__len__() > 0
    print(f"\nDataset length: {dataset.__len__()}")

    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (10, 64, 128, 128)
    assert (~torch.isnan(emb)).any()

    assert isinstance(label, torch.Tensor)
    assert label.shape == (128, 128)
    assert (~torch.isnan(label)).any()
