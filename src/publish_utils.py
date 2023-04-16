#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Yu Hsiang Lo  Final updated time : 2023/04/15
# 此篇的功能為把得到的資訊轉為可用的資料格式

import cv2
import rospy
import numpy as np
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge
import tf

# 定義時間常數與色彩
FARME_ID = 'map'
LIFETIME = 0.1
DETECTION_COLOR_DICT = {'Car':(255,255,0), 'Pedestrian':(0,226,255), 'Cyclist':(140,40,255)}

# 定義 3D box 框線的連接
LINES  = [[0,1], [1,2], [2,3], [3,0]] # lower face 下半部面
LINES += [[4,5], [5,6], [6,7], [7,4]] # upper face 上半部面
LINES += [[4,0], [5,1], [6,2], [7,3]] # connect lower face and upper face 連接下半部和上半部面
LINES += [[4,1], [5,0]] # front face 前面

# 發布相機圖像
def publish_camera(cam_pub, bridge, image, boxes_2d, types):
    for typ, box in zip(types, boxes_2d):
        top_left = int(box[0]),int(box[1])
        bottom_right = int(box[2]),int(box[3])
        cv2.rectangle(image, top_left, bottom_right, DETECTION_COLOR_DICT[typ], 2)
    cam_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))

# 發布點雲資料
def publish_point_cloud(pcl_pub, point_cloud):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = FARME_ID
    pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:, :3]))

# 發布 3D 框線
def publish_3dbox(box3d_pub, corners_3d_velos, types, track_ids):
    marker_array = MarkerArray() # define marker's array 可以放入所有的 array
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        marker = Marker()
        marker.header.frame_id = FARME_ID
        marker.header.stamp = rospy.Time.now()

        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST

        # if types is None:
        #     marker.color.r = 0.0
        #     marker.color.g = 1.0
        #     marker.color.b = 1.0
        # else:
        b, g, r =  DETECTION_COLOR_DICT[types[i]]
        marker.color.r = r/255.0
        marker.color.g = g/255.0
        marker.color.b = b/255.0

        marker.color.a = 1.0
        marker.scale.x = 0.1

        marker.points = [] # define marker.points
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)

        text_marker = Marker()
        text_marker.header.frame_id = FARME_ID
        text_marker.header.stamp = rospy.Time.now()

        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration(LIFETIME)
        text_marker.type = Marker.TEXT_VIEW_FACING

        # 計算中心點
        # p4 = corners_3d_velo[4]
        p = np.mean(corners_3d_velo, axis=0) # center

        text_marker.pose.position.x = p[0]
        text_marker.pose.position.y = p[1]
        text_marker.pose.position.z = p[2] + 1.5 # 提昇z軸高度 才能方便看到完整標記

        # text_marker.text = str(i)
        text_marker.text = str(track_ids[i])

        text_marker.scale.x = 1
        text_marker.scale.y = 1
        text_marker.scale.z = 1
        
        b, g, r =  DETECTION_COLOR_DICT[types[i]]
        text_marker.color.r = r/255.0
        text_marker.color.g = g/255.0
        text_marker.color.b = b/255.0
        text_marker.color.a = 1.0
        marker_array.markers.append(text_marker)

    box3d_pub.publish(marker_array)

# 發布相機視野標記與自體車輛3D模型資訊
def publish_ego_car(ego_car_pub):
    """
    Publish left and right 45 degree FOV lines and ego car model mesh
    """
    # 1. 相機視野標記
    marker_array = MarkerArray()

    marker = Marker()
    marker.header.frame_id = FARME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(0)
    marker.type = Marker.LINE_STRIP

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0    # 透明度
    marker.scale.x = 0.2    # 縮放

    marker.points = [] # define marker.points
    marker.points.append(Point(10, -10, 0))
    marker.points.append(Point(0, 0, 0))
    marker.points.append(Point(10, 10, 0))

    marker_array.markers.append(marker) # 將marker加入marker_array

    # 2. 自體車輛3D模型
    mesh_marker = Marker()
    mesh_marker.header.frame_id = FARME_ID
    mesh_marker.header.stamp = rospy.Time.now()

    mesh_marker.id = -1
    mesh_marker.lifetime = rospy.Duration(0)
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.mesh_resource = "package://kitti_tutorial/bmw_x5/BMWX54.dae"

    mesh_marker.pose.position.x = 0.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -1.73 # 這個是kitti中 感測器設置的高度

    # 旋轉車體角度
    q = tf.transformations.quaternion_from_euler(np.pi/2, 0, np.pi)
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]

    mesh_marker.color.r = 1.0
    mesh_marker.color.g = 1.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

    mesh_marker.scale.x = 0.9
    mesh_marker.scale.y = 0.9
    mesh_marker.scale.z = 0.9

    marker_array.markers.append(mesh_marker)

    ego_car_pub.publish(marker_array)

# 發布IMU資料
def publish_imu(imu_pub, imu_data):
    imu = Imu()
    imu.header.frame_id = FARME_ID
    imu.header.stamp = rospy.Time.now()

    # 歐拉角轉四元數
    q = tf.transformations.quaternion_from_euler(float(imu_data.roll), float(imu_data.pitch), float(imu_data.yaw))
    imu.orientation.x = q[0]
    imu.orientation.y = q[1]
    imu.orientation.z = q[2]
    imu.orientation.w = q[3]
    imu.linear_acceleration.x = imu_data.af
    imu.linear_acceleration.y = imu_data.al
    imu.linear_acceleration.z = imu_data.au
    imu.angular_velocity.x = imu_data.wf
    imu.angular_velocity.y = imu_data.wl
    imu.angular_velocity.z = imu_data.wu

    imu_pub.publish(imu)

# 發布GPS資料
def publish_gps(gps_pub, imu_data):
    gps = NavSatFix()
    gps.header.frame_id = FARME_ID
    gps.header.stamp = rospy.Time.now()

    gps.latitude = imu_data.lat
    gps.longitude = imu_data.lon
    gps.altitude = imu_data.alt

    gps_pub.publish(gps)