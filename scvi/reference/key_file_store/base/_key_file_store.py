from abc import ABC, abstractmethod
from typing import List


class BaseKeyFileStore(ABC):
    @abstractmethod
    def get_all_keys() -> List[str]:
        pass

    def load_file(key: str) -> str:
        pass
