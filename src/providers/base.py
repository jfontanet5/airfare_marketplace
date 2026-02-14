# src/providers/base.py

from abc import ABC, abstractmethod
from typing import List
from core.models import SearchParams, Offer


class FlightSearchProvider(ABC):

    @abstractmethod
    def search(self, params: SearchParams) -> List[Offer]:
        ...
