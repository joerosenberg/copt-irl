from typing import NamedTuple, List, Tuple


class Solution(NamedTuple):
    order: List[int]
    success: bool
    nRouted: int
    measure: float
    pathData: List[List[Tuple[int, int]]]
    failedConnections: List[int]


ProblemInstance = List[Tuple[int, int, int, int]]