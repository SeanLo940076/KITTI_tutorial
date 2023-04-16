#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Yu Hsiang Lo  Final updated time : 2023/04/15         # How to read data
# 這段程式碼中包含了四個函數，分別用於讀取相機圖像、點雲檔案、IMU 資料和追踪資料。這些函數讀取輸入的檔案路徑並將資料載入為適當

import cv2
import numpy as np
import pandas as pd

# 定義 IMU 資料的欄位名稱
IMU_COLUMN_NAMES  = [ 'lat' , 'lon' , 'alt' , 'roll' , 'pitch' , 'yaw' , 'vn' , 've' , 'vf' , 'vl ' , 'vu' , 'ax' , 'ay' , 'az' , 'af' ,
                    'al' , 'au' , 'wx' , 'wy' , 'wz' , 'wf' , 'wl' , 'wu' , 'posacc' , 'velacc' , 'navstat' , 'numsats' , 'posmode ' ,
                    'velmode' , 'orimode' ]

# 定義 TRACKING 文件的欄位名稱
TRACKING_COLUMN_NAMES = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                        'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']

# 讀取相機圖像檔案
def read_camera(path):
    return cv2.imread(path)

# 讀取點雲檔案（3D 座標點資料）
def read_point_cloud(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

# 讀取 IMU（慣性測量單元）資料檔案
def read_imu(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = IMU_COLUMN_NAMES
    return df 

# 讀取追踪資料檔案（包含車輛、行人、自行車等目標的追踪資訊）
def read_tracking(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = TRACKING_COLUMN_NAMES
    df.loc[df.type.isin(['Truck', 'Van', 'Tram']),'type'] = 'Car'
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    return df 