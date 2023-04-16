#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Yu Hsiang Lo  Final updated time : 2023/04/15
# 這段程式碼是用於讀取和處理 KITTI 資料集並將資料發布到 ROS 節點
# 在這部分程式碼中，我們從追踪資料中讀取特定幀(frame)的資料，將 3D 盒子資訊轉換到 Velodyne 座標系，
# 然後讀取相機圖像、點雲和 IMU 資料。最後，我們將這些資料發布到不同的 ROS 主題。
# 程式碼最後使用 rate.sleep() 控制發布速率，並將幀計數器遞增。一旦到達資料集結束，幀計數器將被重置。

# 主程式很簡單 分為兩步驟 1. 建立資料發布的publish 跟建立資料來源路徑  2. 發布資料

import os
from data_utils import *
from publish_utils import *
from kitti_util import *

# 設置資料路徑
DATA_PATH = '/home/sean/Documents/KITTI/2011_09_26_drive_0005_sync/'

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

if __name__ == '__main__':
    # 初始化 ROS 節點
    # Bulid data 
    frame = 0
    rospy.init_node('kitti_node', anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego_car', MarkerArray, queue_size=10)
    imu_pub = rospy.Publisher('kitti_imu', Imu, queue_size=10)
    gps_pub = rospy.Publisher('kitti_gps', NavSatFix, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3d', MarkerArray, queue_size=10)
    bridge = CvBridge()

    rate = rospy.Rate(10)

    # 讀取 TRACKING 資料和 Calibration 資料
    df_tracking = read_tracking('/home/sean/Documents/KITTI/training/label_02/0000.txt')
    calib = Calibration('/home/sean/Documents/KITTI/2011_09_26_calib/2011_09_26/', from_video=True) 

    # 在 ROS 中循環發布資料
    while not rospy.is_shutdown():
        # Read tracking data and pre-processing
        df_tracking_frame = df_tracking[df_tracking.frame==frame]

        types = np.array(df_tracking_frame['type'])
        boxes_2d = np.array(df_tracking_frame[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
        boxes_3d = np.array(df_tracking_frame[['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z','rot_y']])
        track_ids = np.array(df_tracking_frame['track_id'])
        
        corners_3d_velos = [] # define corners_3d_velos
        for box_3d in boxes_3d:
            corners_3d_cam2 = compute_3d_box_cam2(*box_3d)
            corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T) # 8x3 coners array
            corners_3d_velos += [corners_3d_velo]

        # Read raw data
        image = read_camera(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%frame))
        point_cloud = read_point_cloud(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%frame))
        imu_data = read_imu(os.path.join(DATA_PATH, 'oxts/data/%010d.txt'%frame))

        # Publish data 
        publish_camera(cam_pub, bridge, image, boxes_2d, types)
        publish_point_cloud(pcl_pub, point_cloud)
        publish_3dbox(box3d_pub, corners_3d_velos, types, track_ids)
        publish_ego_car(ego_pub)
        publish_imu(imu_pub, imu_data)
        publish_gps(gps_pub, imu_data)

        rospy.loginfo("published frame %d" %frame)
        rate.sleep()

        # frame counter
        frame += 1
        frame %= 154