import importlib
import multiprocessing
from typing import Optional, Sequence, List
import numpy as np
from chromadb.api.types import URI, DataLoader, Image
from concurrent.futures import ThreadPoolExecutor


class NumpyLoader(DataLoader[List[Optional[Image]]]):
    def __init__(self, max_workers: int = multiprocessing.cpu_count()) -> None:
        self._max_workers = max_workers

    def _load_image(self, uri: Optional[np.array]) -> Optional[np.array]:
        return uri if uri is not None else None

    def __call__(self, uris: Sequence[Optional[np.array]]) -> List[Optional[Image]]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self._load_image, uris))