from typing import List
import numpy as np
import torch


def cluster_distance_matrix(d: torch.Tensor, K: int) -> List:
    """ Clustering based on a distance matrix. 

    Steps:
    1. Grouping such that intra-group distance is less than inter-group distance.
    2. Choose a node such that it minimize the mean distance to other nodes in the group.

    Args:
        d: (N, N), d might be upper triangle
        K: number of cluster centers
    
    Returns:
        center_indices: length K list, index of cluster centers.
        clusters: length K list, each contains the index of this group members.
    """
    d = d.clone().cpu().numpy()
    N = len(d)
    num_subgraph = N

    class UnionFind:
        def __init__(self, max_len):
            self.L = [i for i in range(max_len)]
        def find(self, x):
            if self.L[x] == x:
                return x
            self.L[x] = self.find(self.L[x])
            return self.L[x]
        def union(self, x, y):
            self.L[self.find(x)] = self.find(y)
        def finalize(self):
            [self.find(i) for i in range(len(self.L))]
    
    disjoint_sets = UnionFind(N)
    edges = [
        (i, j, d[i][j]) for i in range(N) for j in range(i+1, N)
    ]
    edges = sorted(edges, key=lambda x: x[2])
    while num_subgraph > K:
        i, j, dij = edges.pop(0)
        if disjoint_sets.find(i) != disjoint_sets.find(j):
            num_subgraph -= 1
        disjoint_sets.union(i, j)
    disjoint_sets.finalize()
    
    clusters = dict()
    for i in range(N):
        fi = disjoint_sets.find(i)
        if fi in clusters:
            clusters[fi].append(i)
        else:
            clusters[fi] = [i]
    clusters = list(clusters.values())

    """ Compute cluster center """
    centers = []
    d_full = np.triu(d) + np.triu(d).T
    for cluster in clusters:
        d_sub = d_full[cluster, cluster]
        c = np.argmin(np.mean(d_sub, 0))
        centers.append(cluster[c])
    
    return centers, clusters
        


if __name__ == '__main__':
    def test(N, K):
        np.random.seed(0)
        d = np.random.randint(low=0, high=50, size=[N, N])
        d = np.triu(d)
        print(d)
        res = cluster_distance_matrix(d, K=K)
        print(res)
    test(N=10, K=4)