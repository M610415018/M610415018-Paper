import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from model import FC_Model
from dataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import heapq
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -----------------------------shift----------------------------
def fit_translation(source_points, target_points):
    # 將source_points與target_points作平移fit，並返回平移量
    def cost_function(translation):
        return np.sum((source_points + translation - target_points)**2)
    initial_translation = np.array([0, 0])
    result = minimize(cost_function, initial_translation)
    return 0.9*result.x

# -----------------------------shift----------------------------
# ---------------------------------------------------------Comparison-------------------------------------

GT_array =[]
GPS_array =[]
New_path_array = []

# shift_array = []
path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/GPS_latlon.txt","r+",encoding="UTF-8")
path_GT_f = open("/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/GT_latlon.txt","r+",encoding="UTF-8")

# path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/gpx/Final/shift/shift_test3-5_01.txt","r+",encoding="UTF-8")
# path_GT_f = open("/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/gpx/Final/interp/init/01_GT.txt","r+",encoding="UTF-8")

# path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/gpx/shift/shift_test3-5_01.txt","r+",encoding="UTF-8")
# path_GT_f = open("/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/gpx/init/ITRI_Zhongxing_01_GT.txt","r+",encoding="UTF-8")

# path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/gpx/shift/shift_test3-5_01.txt","r+",encoding="UTF-8")
# path_GT_f = open("/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/gpx/ITRI_Zhongxing_02_GT.txt","r+",encoding="UTF-8")

# path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx/shift_test3-5_01.txt","r+",encoding="UTF-8")
# path_GT_f = open("/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx/0005_GT_lonlat.txt","r+",encoding="UTF-8")

# path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/high/gpx/init/highways_fake.txt","r+",encoding="UTF-8")
# path_GT_f = open("/home/rvl122/paper/dataset/mydataset/high/gpx/init/highways_GT.txt","r+",encoding="UTF-8")

# path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gps.txt","r+",encoding="UTF-8")
# path_GT_f = open("/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gt.txt","r+",encoding="UTF-8")

# path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/gpx/GT/GPS_lonlat_GT.txt","r+",encoding="UTF-8")
# path_GT_f = open("/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/gpx/GT/RTK_lonlat_GT.txt","r+",encoding="UTF-8")


for line in path_GPS_f.readlines():
    line = str(line).strip('\n') 
    GPS_array.append(line)

for line in path_GT_f.readlines():
    line = str(line).strip('\n') 
    GT_array.append(line)
# for line in path_shift_f.readlines():
#     line = str(line).strip('\n') 
#     shift_array.append(line)

GT_array = np.array([list(map(float, s.split(','))) for s in GT_array], dtype='float64')
GPS_array = np.array([list(map(float, s.split(','))) for s in GPS_array], dtype='float64')

 
translation = fit_translation(GT_array, GPS_array)
path_original_translated = GPS_array - translation

# with open("/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx"+'/'+"shift_test3-5_01_o.txt", 'w') as f:
#     for i in range(len(path_original_translated)):
#         f.write("{},{}\n".format(path_original_translated[i][0], path_original_translated[i][1]))
# ---------------------------------------------------------Comparison-------------------------------------

cuda = True
path_front = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/image"
path_seg = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/seg"
path_line = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/line"
path_satellite = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/satellite_512"
path_GPS = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/GPS_latlon.txt"
path_GT = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp//GT_latlon.txt"

# path_front = "/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/image_01"
# path_seg = "/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/seg_01"
# path_line = "/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/14323_01_line"
# path_satellite = "/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/satellite_01_512"
# path_GPS = "/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/gpx/Final/shift/shift_test3-5_01_o.txt"
# path_GT = "/home/rvl122/paper/dataset/mydataset/CCU/14323_all_dataset/gpx/Final/interp/init/01_GT.txt"

# path_front = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/images"
# path_seg = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/seg"
# path_line = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/line"
# path_satellite = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/satellite"
# path_GPS = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/gpx/shift/shift_test3-5_01.txt"
# path_GT = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/gpx/init/ITRI_Zhongxing_01_GT.txt"

# path_front = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/images"
# path_seg = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/seg"
# path_line = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/line"
# path_satellite = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/satellite"
# path_GPS = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/gpx/shift/shift_test3-5_01.txt"
# path_GT = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_02/gpx/ITRI_Zhongxing_02_GT.txt"

# path_front = "/home/rvl122/paper/dataset/mydataset/high/images"
# path_seg = "/home/rvl122/paper/dataset/mydataset/high/seg"
# path_line = "/home/rvl122/paper/dataset/mydataset/high/line"
# path_satellite = "/home/rvl122/paper/dataset/mydataset/high/satellite"
# path_GPS = "/home/rvl122/paper/dataset/mydataset/high/gpx/init/highways_fake.txt"
# path_GT = "/home/rvl122/paper/dataset/mydataset/high/gpx/init/highways_GT.txt"

# path_front = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/image"
# path_seg = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/seg"
# path_line = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/line"
# path_satellite = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/satellite"
# path_GPS = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx/shift_test3-5_01.txt"
# path_GT = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx/0005_GT_lonlat.txt"

# path_front = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/Image"
# path_seg = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/seg"
# path_line = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/line"
# path_satellite = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/satellite"
# path_GPS = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gps.txt"
# path_GT = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gt.txt"

# path_front = "/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/Image"
# path_seg = "/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/seg"
# path_line = "/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/line"
# path_satellite = "/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/satellite"
# path_GPS = "/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/gpx/GT/GPS_lonlat_GT.txt"
# path_GT = "/home/rvl122/paper/dataset/mydataset/ITRI_RTK/2022-11-22-11-14-42_0/gpx/GT/RTK_lonlat_GT.txt"

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

dataset = MyDataset(path_front, path_seg, path_satellite ,path_line, path_GPS, path_GT,transform)
dataloader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=0)
model = FC_Model()
model.to(device)


# model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/CCU/normal_Astar/model_Astar_best.pth", map_location=device)["model_state_dict"])
model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/CCU/shift/model_epoch_best.pth", map_location=device)["model_state_dict"])
# model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/CCU/normal_Astar_change/model_epoch_80.pth", map_location=device)["model_state_dict"])
# model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/KITTI/005/model_epoch_75.pth", map_location=device)["model_state_dict"])
# model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/LPA/model_epoch_78.pth", map_location=device)["model_state_dict"])
# model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/CCU/no_line/model_epoch_81.pth", map_location=device)["model_state_dict"])
# model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/RTK/Japan_1_2.pth", map_location=device)["model_state_dict"])
# model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/junyi/model_epoch_1001.pth", map_location=device)["model_state_dict"])


# txt_file = open('/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/shift/result_test_shift3-5_shift_b.txt', 'w')
# txt_file = open('/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/gpx/shift/result_test_shift3-5_no_shift.txt', 'w')
# txt_file = open('/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/test_no_shift_test.txt', 'w')
txt_file = open('/home/rvl122/Desktop/123.txt','w')


def SE(forecast, prediction):
    yerror = ((forecast[0]-prediction[0])*110936.2)**2
    xerror = ((forecast[1]-prediction[1])*101775.45)**2
    SE = (xerror+yerror)/2
    return SE

def cat_dist(gt, pred):
    #####MAE####
    d_y = abs(gt[:,0]- pred[:,0]) * 110936.2 
    d_x = abs(gt[:,1]- pred[:,1]) * 101775.45  

    dist_x = np.mean(d_x)
    dist_y = np.mean(d_y)
    MAE = round((dist_x + dist_y), 4)

    #####RMSE####
    MSE = 0
    for i in range(len(gt)):
        # MAE = 
        se = SE(gt[i], pred[i])
        MSE += se 
        # print(RMSE_np)
    MSE = (MSE/len(gt) )# * length
    RMSE = round(math.sqrt(MSE), 4)

    #####short distance####
    dist = cdist(gt, pred, 'euclidean')
    min_dist = np.min(dist, axis=1)
    meter = min_dist * 111194.9
    short_dist = np.mean(meter)
    return MAE, RMSE,short_dist

for i, (images_front, images_seg, images_satellite, image_line, GPS, target) in enumerate(dataloader):
    print(i)
    with torch.no_grad():
        output = model(images_front,images_seg,images_satellite,image_line)   # shape of output = (B, 2); (delta_x, delta_y)

# Save the output to a file
    delta_x, delta_y = output.squeeze().tolist()
    txt_file.write('{},{}\n'.format((delta_x/1000)+GPS[0][0], (delta_y/1000)+GPS[0][1]))

    new_lon = ((delta_x/1000)+GPS[0][0]).item()
    new_lat = ((delta_y/1000)+GPS[0][1]).item()
    New_path_array.append([new_lon,new_lat])

New_path_array = np.array(New_path_array, dtype='float64')

shift_MAE,shift_RMSE,shift_short_distance =cat_dist(GT_array,path_original_translated)
old_MAE,old_RMSE,old_short_distance = cat_dist(GPS_array,GT_array)
new_MAE,new_RMSE,new_short_distance = cat_dist(New_path_array,GT_array)

print("old_MAE:{:.2f}m".format(old_MAE))
print("new_MAE:{:.2f}m".format(new_MAE))
print("shift_MAE:{:.2f}m".format(shift_MAE))
print("Improvement MAE rate:{:.2f}%".format((old_MAE-new_MAE)/old_MAE*100))
print("Improvement shift_MAE rate:{:.2f}%".format((old_MAE-shift_MAE)/old_MAE*100))
print("old_short_distance:{:.2f}m".format(old_short_distance))
print("new_short_distance:{:.2f}m".format(new_short_distance))
print("shift_short_distance:{:.2f}m".format(shift_short_distance))
print("Improvement shortdistance rate:{:.2f}%".format((old_short_distance-new_short_distance)/old_short_distance*100))
print("Improvement shift_short_distance rate:{:.2f}%".format((old_short_distance-shift_short_distance)/old_short_distance*100))
# print("old_RMSE:{:.2f}m".format(old_RMSE))
# print("new_RMSE:{:.2f}m".format(new_RMSE))
# print("shift_RMSE:{:.2f}m".format(shift_RMSE))
# print("Improvement RMSE rate:{:.2f}%".format((old_RMSE-new_RMSE)/old_RMSE*100))
# print("Improvement shift_RMSE rate:{:.2f}%".format((old_RMSE-shift_RMSE)/old_RMSE*100))
# plt.plot(GT_array[:, 0], GT_array[:, 1], '.', color='orange', label='GT')
# plt.plot(GPS_array[:, 0], GPS_array[:, 1], '.', color='red', label='Raw')
# plt.plot(path_original_translated[:, 0], path_original_translated[:, 1], '.', label='Translate')
# plt.plot(New_path_array[:, 0], New_path_array[:, 1], '.',color='turquoise',  label='A star')
# plt.legend()
# plt.show()
txt_file.close()
