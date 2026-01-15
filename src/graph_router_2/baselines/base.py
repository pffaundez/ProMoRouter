from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional

from .types import RoutingDecision

class BaselineRouter(ABC):
    """Common for all routers"""

    def __init__(self, name: str, default_prompt_id: Optional[str] = None):
        self.name = name
        self.default_prompt_id = default_prompt_id

    @abstractmethod
    def route_one(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """Route a single query or sub-query (if div = 1 -> q = sq)"""
        raise NotImplementedError

    def route_batch(self, queries: list[str], context: Optional[Dict] = None) -> list[RoutingDecision]:
        return [self.route_one(q, context=context) for q in queries]
