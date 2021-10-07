from typing import Type

import pandas as pd

from scvi.model.base import BaseModelClass
from scvi.reference.base import BaseReference


class TabulaSapiensReference(BaseReference):
    def get_available_reference_models(self) -> pd.DataFrame:
        pass

    def load_reference_model(
        self, model_id: str, load_anndata: bool
    ) -> Type[BaseModelClass]:
        pass
