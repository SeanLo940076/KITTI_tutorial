#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Yu Hsiang Lo  Final updated time : 2023/04/15
# 這段程式碼是用於讀取和處理 KITTI 資料集並將資料發布到 ROS 節點
# 在這部分程式碼中，我們從追踪資料中讀取特定幀(frame)的資料，將 3D 盒子資訊轉換到 Velodyne 座標系，
# 然後讀取相機圖像、點雲和 IMU 資料。最後，我們將這些資料發布到不同的 ROS 主題。
# 程式碼最後使用 rate.sleep() 控制發布速率，並將幀計數器遞增。一旦到達資料集結束，幀計數器將被重置。

# 主程式很簡單 分為兩步驟 1. 建立資料發布的publish 跟建立資料來源路徑  2. 發布資料

import os
from collections import deque # google deque
from data_utils import *
from publish_utils import *
from kitti_util import *
from misc import *

# 設置資料路徑
DATA_PATH = '/home/sean/Documents/KITTI/2011_09_26_drive_0005_sync/'
EGOCAR = np.array([[2.15, 0.9, -1.73], [2.15, -0.9, -1.73], [-1.95, -0.9, -1.73], [-1.95, 0.9, -1.73],
                   [2.15, 0.9, -0.23], [2.15, -0.9, -0.23], [-1.95, -0.9, -0.23], [-1.95, 0.9, -0.23]])

class Object():
    def __init__(self, center):
        self.locations = deque(maxlen=20) # 保留最近20幀就好
        self.locations.appendleft(center) # 

    def update(self, center, displacement, yaw_change):
        for i in range(len(self.locations)):
            x0, y0 = self.locations[i]
            x1 = x0 * np.cos(yaw_change) + y0 * np.sin(yaw_change) - displacement # x1 是新的x座標
            y1 = -x0 * np.sin(yaw_change) + y0 * np.cos(yaw_change)
            self.locations[i] = np.array([x1, y1])

        if center is not None: # 因為有些物件在當前這一幀沒有被偵測到，所以會出現None
            self.locations.appendleft(center) # 現在不只有自己的車了，還有其他的物體
        # (舊的)self.locations.appendleft(np.array([0, 0])) # 第一幀永遠是(0, 0) 並且因為是自己的車，所以後面要加上的每一幀都是(0, 0)

    def reset(self):
        self.locations = deque(maxlen=20)


if __name__ == '__main__':
    # 初始化 ROS 節點
    # First Bulid data 
    frame = 0
    rospy.init_node('kitti_node', anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego_car', MarkerArray, queue_size=10)
    imu_pub = rospy.Publisher('kitti_imu', Imu, queue_size=10)
    gps_pub = rospy.Publisher('kitti_gps', NavSatFix, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3d', MarkerArray, queue_size=10)
    loc_pub = rospy.Publisher('kitti_loc', MarkerArray, queue_size=10)
    dist_pub = rospy.Publisher('kitti_dist', MarkerArray, queue_size=10)
    other3D_pub = rospy.Publisher('kitti_other3D', MarkerArray, queue_size=10)
    bridge = CvBridge()

    rate = rospy.Rate(10)

    # 讀取 TRACKING 資料和 Calibration 資料
    df_tracking = read_tracking('/home/sean/Documents/KITTI/training/label_02/0000.txt')
    calib = Calibration('/home/sean/Documents/KITTI/2011_09_26_calib/2011_09_26/', from_video=True) 

    # ego_car = Object() # 不再是只有自身車體，所以更改為 tracker = {}
    # tracker = {} 和 centers = {} 的差異，
    # tracker 是要紀錄所有追蹤的物體，包括過去到現在的這些物體，因為可能物體會被遮擋
    # centers 是只有紀錄當前這一幀的所有偵測到的物體
    tracker = {} # track_id : Object  
    prev_imu_data = None

    # 在 ROS 中循環發布資料
    while not rospy.is_shutdown():
        # Read tracking data and pre-processing
        df_tracking_frame = df_tracking[df_tracking.frame==frame]

        types = np.array(df_tracking_frame['type'])
        boxes_2d = np.array(df_tracking_frame[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
        boxes_3d = np.array(df_tracking_frame[['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z','rot_y']])
        track_ids = np.array(df_tracking_frame['track_id'])
        
        corners_3d_velos = [] # define corners_3d_velos
        centers = {} # track_id : center 
        minPQDs = [] # 那兩個點P, Q ，和最短距離D是多少

        for track_id, box_3d in zip(track_ids, boxes_3d): # 要知道3D物件的 track_id (每一個物體的ID) 才能
            corners_3d_cam2 = compute_3d_box_cam2(*box_3d)
            corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T) # 8x3 coners array
            # minPQDs 這邊的程式代表著，對於所有物體都計算出他們與自身車體的距離和那兩個點，並且紀錄在list內
            minPQDs += [min_distance_cuboids(EGOCAR, corners_3d_velo)] # 計算距離的發法寫在 min_distance_cuboids
            corners_3d_velos += [corners_3d_velo]
            centers[track_id] = np.mean(corners_3d_velo, axis=0)[:2] # 計算8頂點的平均(x,y,z)，接下來放棄z值 axis=0代表 垂直方向取平均 ｜ 指考慮鳥瞰圖方向的中心點 所以z軸 don't care
        # 設定自身車輛
        corners_3d_velos += [EGOCAR]
        types = np.append(types,'Car')
        track_ids = np.append(track_ids, -1)
        centers[-1] = np.array([0, 0]) # 把自身車體也當成其他物體來看待，但自身都是在(0, 0) 所以位置就設在(0, 0)即可，自身車輛的軌跡就回來了

        # Read raw data
        # %010d 是一个占位符，用于将 frame 转换为一个 10 位的零填充整数。例如，如果 frame 是 42，那么生成的文件名将是 0000000042.png。
        image = read_camera(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%frame)) 
        point_cloud = read_point_cloud(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%frame))
        imu_data = read_imu(os.path.join(DATA_PATH, 'oxts/data/%010d.txt'%frame))

        if prev_imu_data is None:       # 當並沒有前一幀的時候(也就是第一幀時)，我們對所有物體都要先建立一個物件，之後才能對這些物件進行更新
            for track_id in centers:    
                tracker[track_id] = Object(centers[track_id])    # 所以新建一個Object給它
        else:
            # (此註解為舊版本才需要看 之前的程式碼是 使用"+=" 來實現 append的效果)注意 這裡也是個list  # 這裡的 0.1是因為幀數的關係
            # 首先計算等等會用到的參數，移動量, 角度變化
            displacement = 0.1 * np.linalg.norm(imu_data[['vf', 'vl']]) 
            yaw_change = float(imu_data.yaw - prev_imu_data.yaw)
            
            for track_id in centers:
                # 情境1. 當下偵測到物體的是"已被偵測過的"
                if track_id in tracker:
                    # 將物件位置進行更新，並且加上當前偵測到的中心 
                    tracker[track_id].update(centers[track_id], displacement, yaw_change)

                # 情境2. 當下偵測到的物體是以前"沒被偵測過的"
                else:
                    tracker[track_id] = Object(centers[track_id])    # 所以新建一個Object給它
            # 在上述的物體中
            for track_id in tracker:
                # 過去有被追蹤到，但這一幀可能被遮擋，沒有偵測到，我們也一樣要進行更新
                if track_id not in centers:
                    # 將物件進行更新過去軌跡，但因為不知到當前的中心在那，所以給 None 
                    tracker[track_id].update(None, displacement, yaw_change)   

            # (舊的)ego_car.update(displacement, yaw_change) # 更新自身車體的距離跟角度

        prev_imu_data = imu_data

        # Publish data 
        publish_camera(cam_pub, bridge, image, boxes_2d, types)
        publish_point_cloud(pcl_pub, point_cloud)
        publish_3dbox(box3d_pub, corners_3d_velos, types, track_ids)
        publish_ego_car(ego_pub)
        publish_imu(imu_pub, imu_data)
        publish_gps(gps_pub, imu_data)
        # (舊的)publish_loc(loc_pub, ego_car.locations)
        publish_loc(loc_pub, tracker, centers)
        publish_dist(dist_pub, minPQDs)
        publish_other3D(other3D_pub, tracker, centers, types)
        # publish_other3D(other3D_pub, corners_3d_velos, types, track_ids)
        rospy.loginfo("published frame %d" %frame)
        rate.sleep()

        # frame counter
        frame += 1
        if frame == 154:
            frame = 0
            for track_id in tracker:
                tracker[track_id].reset()


        