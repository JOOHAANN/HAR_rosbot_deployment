import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# COCO-17 keypoints (RTMPose output order):
# 1 nose, 2 left_eye, 3 right_eye, 4 left_ear, 5 right_ear,
# 6 left_shoulder, 7 right_shoulder, 8 left_elbow, 9 right_elbow,
# 10 left_wrist, 11 right_wrist, 12 left_hip, 13 right_hip,
# 14 left_knee, 15 right_knee, 16 left_ankle, 17 right_ankle
num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(2, 1), (3, 1), (4, 2), (5, 3),
                    (6, 1), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                    (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
