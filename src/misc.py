#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Yu Hsiang Lo  Final updated time : 2023/04/15

import numpy as np

# 計算 3D box 在 cam2 坐標系中的位置
def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn (n=8) in cam2 coordinate
    """
    # R 是旋轉矩陣 (此為繞y軸旋轉)
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],[0, 1, 0], [-np.sin(yaw), 0,np.cos(yaw)]])
    # Ref https://github.com/pratikac/kitti/blob/master/readme.tracking.txt
    x_corners = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [0,   0,    0,    0,   -h,   -h,   -h,   -h  ]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2  ]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2

def distance_point_to_segment(P,A,B):
    """
    Calculates the min distance of a point P to a segment AB.
    Returns the point Q in AB on which the min distance is reached.
    """
    AP = P - A
    BP = P - B
    AB = B - A
    if np.dot(AB, AP) >= 0 and np.dot(-AB, BP) >= 0:
        return np.abs(np.cross(AP, AB)) / np.linalg.norm(AB), np.dot(AP, AB) / np.dot(AB, AB) * AB + A
    d_PA = np.linalg.norm(AP)
    d_PB = np.linalg.norm(BP)
    if d_PA < d_PB:
        return d_PA, A
    return d_PB, B

def min_distance_cuboids(cub1, cub2):
    """
    Computes the minimum distance between two non-overlapping cuboids (3D) of shape(8, 3).
    They are projected to BEV and the minmum distance of the two rectangles are returned.
    """
    minD = 1e5 # 一開始設一個很大的值
    for i in range(4):
        for j in range(4): # distance_point_to_segment(P, A, B)
            # 設R1 上的P 到R2 上的AB邊長 
            d, Q = distance_point_to_segment(cub1[i, :2], cub2[j, :2], cub2[j+1, :2])
            if d < minD:
                minD = d
                minP = cub1[i, :2]
                minQ = Q
        # 交換 設R2 上的P 到R1 上的AB邊長
        for j in range(4):
            d, Q = distance_point_to_segment(cub2[i, :2], cub1[j, :2], cub1[j+1, :2])
            if d < minD:
                minD = d
                minP = cub2[i, :2]
                minQ = Q
    return minP, minQ, minD