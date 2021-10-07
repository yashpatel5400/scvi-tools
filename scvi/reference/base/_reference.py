from abc import ABC, abstractmethod
from typing import Type

import pandas as pd

from scvi.model.base import BaseModelClass


class BaseReference(ABC):
    @abstractmethod
    def get_available_reference_models(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_reference_model(
        self, model_id: str, load_anndata: bool = False
    ) -> Type[BaseModelClass]:
        pass
