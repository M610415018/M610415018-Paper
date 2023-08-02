# M610415018-Paper
## Dataset
[download](https://drive.google.com/file/d/1wiChiNabzU3tX0_hB_rxyuWfk6ad71PS/view?usp=sharing)

## Requirement
crate anconda requirement 
hc => main_code
seg => seg_code

## Paper Operation
Pre-actions for data processing
1.gpx info to latlon info  =>   ~/home/rvl122/paper/dataset/mydataset/str_split.py
2.video to image  =>  ~/home/rvl122/paper/dataset/mydataset/extract_frame.py
3.same gps and image  =>  ~/home/rvl122/paper/dataset/mydataset/gps_interp2d.py
4.if you want use trajectory shift location info => ~/home/rvl122/paper/dataset/mydataset/trajectory_shift.py

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