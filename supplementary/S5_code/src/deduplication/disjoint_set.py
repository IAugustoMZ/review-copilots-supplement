"""Minimal union-find so pairwise match edges (doi_exact / title_exact /
title_fuzzy, from possibly different comparisons) collapse transitively into
duplicate groups without a separate graph library dependency."""

from __future__ import annotations


class DisjointSet:
    def __init__(self, ids: list[str]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # deterministic root choice so repeated runs over the same data agree
            if root_b < root_a:
                root_a, root_b = root_b, root_a
            self._parent[root_b] = root_a

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for member in self._parent:
            root = self.find(member)
            result.setdefault(root, []).append(member)
        return result
