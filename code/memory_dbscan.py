"""Exact unweighted Euclidean DBSCAN without storing the full radius graph."""

import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


class MemoryBoundDBSCAN:
    """Certify core components using kNN edges, then complete missing edges."""

    def __init__(self, x, max_min_samples=100, query_batch=128):
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        if self.x.ndim != 2 or not len(self.x) or not np.isfinite(self.x).all():
            raise ValueError("Expected a nonempty finite feature matrix")
        self.batch = int(query_batch)
        self.k = min(int(max_min_samples), len(self.x))
        if self.batch < 1 or self.k < 1:
            raise ValueError("Batch size and maximum min_samples must be positive")
        self.nn = NearestNeighbors(algorithm="brute", metric="euclidean", n_jobs=1).fit(self.x)
        self.distances = np.empty((len(self.x), self.k), dtype=np.float64)
        self.indices = np.empty((len(self.x), self.k), dtype=np.int32)
        started = time.perf_counter()
        for start in range(0, len(self.x), self.batch):
            end = min(start + self.batch, len(self.x))
            d, i = self.nn.kneighbors(self.x[start:end], n_neighbors=self.k)
            self.distances[start:end], self.indices[start:end] = d, i
            if end % 5000 < self.batch or end == len(self.x):
                print(f"kNN cache: {end:,}/{len(self.x):,}", flush=True)
        self.precompute_seconds = time.perf_counter() - started

    def fit_predict(self, eps, min_samples):
        n = len(self.x)
        if eps <= 0 or min_samples < 1:
            raise ValueError("eps and min_samples must be positive")
        if min_samples > n:
            return np.full(n, -1, dtype=np.int32), np.zeros(n, dtype=bool)
        if min_samples > self.k:
            raise ValueError("min_samples exceeds the cached nearest-neighbor budget")
        core = self.distances[:, min_samples - 1] <= eps
        core_ids = np.flatnonzero(core)
        labels = np.full(n, -1, dtype=np.int32)
        if not len(core_ids):
            return labels, core

        # kNN edges are a certified subset of the complete epsilon core graph.
        near = self.indices[core_ids]
        keep = (self.distances[core_ids] <= eps) & core[near]
        rows, slots = np.nonzero(keep)
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (core_ids[rows], near[rows, slots])),
            shape=(n, n),
        )
        _, graph_labels = connected_components(graph, directed=False)
        _, core_components = np.unique(graph_labels[core_ids], return_inverse=True)
        del graph, graph_labels, near, keep, rows, slots
        component_of = np.full(n, -1, dtype=np.int32)
        component_of[core_ids] = core_components
        sizes = np.bincount(core_components)
        count = len(sizes)
        parents = np.arange(count, dtype=np.int32)
        ranks = np.zeros(count, dtype=np.int8)
        remaining = count

        def root(item):
            item = int(item)
            while parents[item] != item:
                parents[item] = parents[parents[item]]
                item = int(parents[item])
            return item

        def merge(a, b):
            nonlocal remaining
            a, b = root(a), root(b)
            if a == b:
                return
            if ranks[a] < ranks[b]:
                a, b = b, a
            parents[b] = a
            if ranks[a] == ranks[b]:
                ranks[a] += 1
            remaining -= 1

        if count > 1:
            lower = np.full((count, self.x.shape[1]), np.inf)
            upper = np.full_like(lower, -np.inf)
            for dimension in range(self.x.shape[1]):
                np.minimum.at(lower[:, dimension], core_components, self.x[core_ids, dimension])
                np.maximum.at(upper[:, dimension], core_components, self.x[core_ids, dimension])
            largest = int(sizes.argmax())
            # Every intercomponent edge has an endpoint outside this largest component.
            for component in np.argsort(sizes, kind="stable"):
                component = int(component)
                if component == largest or remaining == 1:
                    continue
                gaps = np.maximum(np.maximum(lower[component] - upper, lower - upper[component]), 0)
                roots = np.array([root(i) for i in range(count)])
                # Only prune certified separation; radius queries decide near-boundary edges.
                tolerance = 64 * np.finfo(float).eps * max(1.0, eps * eps)
                possible = (np.einsum("ij,ij->i", gaps, gaps) <= eps * eps + tolerance) & (roots != root(component))
                if not possible.any():
                    continue
                ids = core_ids[core_components == component]
                for start in range(0, len(ids), self.batch):
                    neighbors = self.nn.radius_neighbors(
                        self.x[ids[start:start + self.batch]], radius=eps, return_distance=False
                    )
                    for neighborhood in neighbors:
                        adjacent = np.unique(component_of[neighborhood])
                        for other in adjacent[adjacent >= 0]:
                            merge(component, int(other))
                    if remaining == 1:
                        break

        roots = np.array([root(i) for i in range(count)])
        core_roots = roots[core_components]
        first = np.full(count, n, dtype=np.int64)
        np.minimum.at(first, core_roots, core_ids)
        ordered_roots = np.flatnonzero(first < n)
        ordered_roots = ordered_roots[np.argsort(first[ordered_roots], kind="stable")]
        root_label = np.full(count, -1, dtype=np.int32)
        root_label[ordered_roots] = np.arange(len(ordered_roots))
        labels[core_ids] = root_label[core_roots]

        # sklearn visits components by their first core index; borders use the first one.
        noncore = np.flatnonzero(~core)
        if len(noncore):
            core_nn = NearestNeighbors(algorithm="brute", metric="euclidean", n_jobs=1).fit(self.x[core_ids])
            for start in range(0, len(noncore), self.batch):
                ids = noncore[start:start + self.batch]
                neighbors = core_nn.radius_neighbors(self.x[ids], radius=eps, return_distance=False)
                for index, neighborhood in zip(ids, neighbors):
                    if len(neighborhood):
                        labels[index] = labels[core_ids[neighborhood]].min()
        return labels, core
