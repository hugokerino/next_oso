from pathlib import Path
from typing import Optional, List
import torch.utils.data as tdata


class BaseDataset(tdata.Dataset):
    VALID_LABEL_TYPES = {
        "max_class",
        "oso_type",
        "probas",
        "extended",
        "probas_extended",
        None,
    }
    VALID_FOLD_RANGE = (1, 5)

    def __init__(
        self,
        path_data: Path,
        folds: list[int] | None,
        label_type: str | None,
        return_id: bool,
    ):
        """
        Base class for dataset creation, intended to mutualize shared code.

        Parameters
        ----------
        path_data : Path
            Path to the directory containing the dataset.
        folds : list[int] | None
            Dataset folds to load. If None, all available folds are used.
            Values must be within VALID_FOLD_RANGE.
        label_type : str | None
            Type of label to return. Must be one of VALID_LABEL_TYPES.
            None should be use only for embedding dataset on PASTIS.
        return_id : bool
            Whether to return the sample identifier together with the data and labels.

        """

        super().__init__()

        if not isinstance(path_data, Path):
            path_data = Path(path_data)
        if not path_data.exists():
            raise FileNotFoundError(f"Path {path_data} does not exist.")

        self.path_data = path_data
        self.folds = self._validate_folds(folds)
        self.label_type = self._validate_label_type(label_type)
        self.return_id = return_id

    def _validate_folds(self, folds: Optional[List[int]]) -> Optional[List[int]]:
        if folds is None:
            return None

        # Check if folds are int
        if not all(isinstance(f, int) for f in folds):
            raise TypeError(f"All fold values must be integers. Got {folds}")

        # Check range for folds
        min_fold, max_fold = self.VALID_FOLD_RANGE
        if not all(min_fold <= f <= max_fold for f in folds):
            raise ValueError(
                f"Fold values must be between {min_fold} and {max_fold}. Got {folds}"
            )

        return folds

    def _validate_label_type(self, label_type: str) -> str:
        if label_type not in self.VALID_LABEL_TYPES:
            raise ValueError(
                f"label_type must be one of {self.VALID_LABEL_TYPES}. Got {label_type}"
            )
        return label_type
