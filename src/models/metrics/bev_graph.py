import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.sparse import csgraph

from pytorch_helper.utils.io import config

sns.set_theme(style='dark')


class BEVGraph(object):
    def __init__(self, coords, min_dist=2, n=2, soft_weight=False):
        assert coords.ndim == 2 and coords.shape[1] == 2, \
            'invalid coordinates tensor'
        self.coords = coords
        self.n_annos = coords.shape[0]
        self.min_dist = min_dist
        self.pair_dist = self.get_pair_distance(self.coords)
        self.active_pairs = self.get_active_pairs(self.pair_dist, self.min_dist)
        if soft_weight:
            self.adj = self._build_graph(self.coords, self.active_pairs,
                                         1 / self.pair_dist)
        else:
            self.adj = self._build_graph(self.coords, self.active_pairs)
        self.graph = (self.coords, self.adj)
        self.clusters = self.get_active_clusters(n)
        self.individuals = self.get_active_individual()

    def savefig(self, path, return_nodes=True, return_edges=True,
                return_clusters=True
                ):
        _, ax = plt.subplots(1)
        if return_nodes:
            ax.scatter(-self.coords[:, 1], self.coords[:, 0])
            for k in range(self.n_annos):
                ax.annotate('', (-self.coords[k, 1], self.coords[k, 0]))
        if return_edges:
            for i, j in self.active_pairs:
                x0, y0 = self.coords[i]
                x1, y1 = self.coords[j]
                ax.plot((-y0, -y1), (x0, x1), color='0.5')
        if return_clusters:
            for bbox_min, bbox_max in self.get_cluster_bboxes():
                w, h = bbox_max - bbox_min
                rect = patches.Rectangle(
                    (-bbox_min[1] - h, bbox_min[0]), h, w,
                    linewidth=1, ec='r', fc=(1, 0, 0, 0.2)
                )
                ax.add_patch(rect)

        plt.axis('off')
        plt.savefig(f'{path}.{config["img_ext"]}', bbox_inches='tight',
                    pad_inches=0)
        plt.close()

    @staticmethod
    def get_pair_distance(coord):
        """
        returns pairwise distance between coordinates
        coord: (N, D) tensor
        """
        return torch.norm(coord.unsqueeze(0) - coord.unsqueeze(1), dim=2)

    @staticmethod
    def get_active_pairs(pair_dist, min_dist):
        """
        returns pairs of individuals with distance < min_distance
        pair_dist: (N, N) tensor
        """
        pairs = torch.nonzero(
            (pair_dist > 0) & (pair_dist < min_dist), as_tuple=False
        )
        return pairs[pairs[:, 0] < pairs[:, 1]]

    def get_active_individual(self):
        """
        returns indices of individuals with distance < min_distance from n others
        """
        n_edges = csgraph.csgraph_to_masked(self.adj).sum(1)
        indices, = n_edges.nonzero()
        return indices

    def get_active_clusters(self, n):
        """
        returns clusters of individuals with distance < min_distance
        i.e. connected component of graph, with active pairs as edges
        """
        n_comp, comp_label = csgraph.connected_components(self.adj)
        components = []
        for i in range(n_comp):
            if (comp_label == i).sum() >= n:
                index, = (comp_label == i).nonzero()
                components.append(index)
        return components

    def get_cluster_bboxes(self):
        bboxes = []
        for i in range(len(self.clusters)):
            cluster_coords = self.coords[self.clusters[i]]
            bbox_min, _ = cluster_coords.min(0)
            bbox_max, _ = cluster_coords.max(0)
            bboxes.append((bbox_min, bbox_max))
        return bboxes

    @staticmethod
    def _build_graph(nodes, edges, weight=None):
        # adjacency matrix
        adj = np.full((len(nodes), len(nodes)), np.nan)
        for i, j in edges:
            w = 1 if weight is None else weight[i, j]
            adj[i, j] = w
            adj[j, i] = w

        # build graph
        return csgraph.csgraph_from_dense(adj)
