crate anconda requirement hc => main_code
                          seg => seg_code


paper operation
------------------------------------------------------------------------------
Pre-actions for data processing
1.gpx info to latlon info  =>   ~/home/rvl122/paper/dataset/mydataset/str_split.py
2.video to image  =>  ~/home/rvl122/paper/dataset/mydataset/extract_frame.py
3.same gps and image  =>  ~/home/rvl122/paper/dataset/mydataset/gps_interp2d.py
4.if you want use trajectory shift location info => ~/home/rvl122/paper/dataset/mydataset/trajectory_shift.py
------------------------------------------------------------------------------
input image info lane line,seg,satellite
1.lane line and seg
training lane line and seg
~/home/rvl122/paper/BiSeNet-ooooverflow/train.py
change (line495)~(line501)

lane line
step1 : open ~/home/rvl122/paper/BiSeNet-ooooverflow/lane_demo.py
step2 : change input and output => (line225),(line232) 

seg
step1 : open ~/home/rvl122/paper/BiSeNet-ooooverflow/demo.py
step2 : change input and output => (line261),(line270)

maybe you will use resort 
/home/rvl122/paper/BiSeNet-ooooverflow/resort.py
------------------------------------------------------------------------------
2.satellite
step 1: Download QGIS and Plugins install Lat Lon Tools => (can change Coordinate in setting)
step 2: Input satellite image.tif
step 3: Capture satellite => ~/home/rvl122/paper/dataset/mydataset/epsg2png.py
------------------------------------------------------------------------------

main code operation
------------------------------------------------------------------------------
training 
~/home/rvl122/paper/main/train_Astar.py

#save weight path
(line77)   save_path=(" ")

#change input
path_front = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/Image"
path_seg = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/seg"
path_line = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/line"
path_satellite = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/satellite"
path_GPS = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gps.txt"
path_GT = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gt.txt"


lr = 1e-3
batch_size = 1
Init_Epoch = 0
Fin_Epoch = 100

# pre train =>
if pre_train:
model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/RTK/Japan_1_2.pth", map_location=device)["model_state_dict"])
------------------------------------------------------------------------------
predict
~/home/rvl122/paper/main/pred.py

#LSM function
def fit_translation(source_points, target_points):
    # 將source_points與target_points作平移fit，並返回平移量
    def cost_function(translation):
        return np.sum((source_points + translation - target_points)**2)
    initial_translation = np.array([0, 0])
    result = minimize(cost_function, initial_translation)
    return 0.9*result.x

path_GPS_f = open("/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/GPS_latlon.txt","r+",encoding="UTF-8")
path_GT_f = open("/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/GT_latlon.txt","r+",encoding="UTF-8")

#save new LSM trajectory 
with open("/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx"+'/'+"shift_test3-5_01_o.txt", 'w') as f:
    for i in range(len(path_original_translated)):
        f.write("{},{}\n".format(path_original_translated[i][0], path_original_translated[i][1]))


#change input
path_front = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/image"
path_seg = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/seg"
path_line = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/line"
path_satellite = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/satellite_512"
path_GPS = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/GPS_latlon.txt"
path_GT = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp//GT_latlon.txt"

#change weight
model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/CCU/shift/model_epoch_best.pth", map_location=device)["model_state_dict"])

#save result
txt_file = open('/home/rvl122/Desktop/123.txt','w')


