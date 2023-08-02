Crate anconda requirement 
--------------------------------------------------------------------------------
hc => main_code <br>
seg => seg_code <br>

Paper Operation
--------------------------------------------------------------------------------
Pre-actions for data processing <br>
1.gpx info to latlon info  =>   ~/home/rvl122/paper/dataset/mydataset/str_split.py <br>
2.video to image  =>  ~/home/rvl122/paper/dataset/mydataset/extract_frame.py <br>
3.same gps and image  =>  ~/home/rvl122/paper/dataset/mydataset/gps_interp2d.py <br>
4.if you want use trajectory shift location info => ~/home/rvl122/paper/dataset/mydataset/trajectory_shift.py <br>

input image info lane line,seg,satellite <br>
1.lane line and seg <br>
training lane line and seg <br>
/home/rvl122/paper/BiSeNet-ooooverflow/train.py <br>
change (line495)~(line501) <br>

lane line <br>
step1 : open ~/home/rvl122/paper/BiSeNet-ooooverflow/lane_demo.py <br>
step2 : change input and output => (line225),(line232)  <br>

seg <br>
step1 : open ~/home/rvl122/paper/BiSeNet-ooooverflow/demo.py <br>
step2 : change input and output => (line261),(line270) <br>
maybe you will use resort  <br>
/home/rvl122/paper/BiSeNet-ooooverflow/resort.py <br>

2.satellite <br>
step 1: Download QGIS and Plugins install Lat Lon Tools => (can change Coordinate in setting) <br>
step 2: Input satellite image.tif <br>
step 3: Capture satellite => ~/home/rvl122/paper/dataset/mydataset/epsg2png.py <br>

main code operation
--------------------------------------------------------------------
training  <br>
~/home/rvl122/paper/main/train_Astar.py <br>

#save weight path <br>
(line77)   save_path=(" ") <br>

#change input <br>
path_front = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/Image" <br>
path_seg = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/seg" <br>
path_line = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/line" <br>
path_satellite = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/satellite" <br>
path_GPS = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gps.txt" <br>
path_GT = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gt.txt" <br>


lr = 1e-3 <br>
batch_size = 1 <br>
Init_Epoch = 0 <br>
Fin_Epoch = 100 <br>

##pre train => <br>
if pre_train: <br>
model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/RTK/Japan_1_2.pth", map_location=device)["model_state_dict"]) <br>

predict <br>
~/home/rvl122/paper/main/pred.py <br>

change input <br>

path_front = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/image" <br>
path_seg = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/seg" <br>
path_line = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/line" <br>
path_satellite = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/satellite_512" <br>
path_GPS = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/GPS_latlon.txt" <br>
path_GT = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp//GT_latlon.txt" <br>

#change weight <br>
model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/CCU/shift/model_epoch_best.pth", map_location=device)["model_state_dict"]) <br>

#save result <br>
txt_file = open('/home/rvl122/Desktop/123.txt','w') <br>


