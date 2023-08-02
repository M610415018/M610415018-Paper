# M610415018-Paper
## Dataset
[download](https://drive.google.com/file/d/1wiChiNabzU3tX0_hB_rxyuWfk6ad71PS/view?usp=sharing)

## Requirement
crate anconda requirement <br>
hc => main_code <br>
seg => seg_code <br>

## Paper Operation
Pre-actions for data processing
1.gpx info to latlon info  =>   ~/home/rvl122/paper/dataset/mydataset/str_split.py <br>
2.video to image  =>  ~/home/rvl122/paper/dataset/mydataset/extract_frame.py <br>
3.same gps and image  =>  ~/home/rvl122/paper/dataset/mydataset/gps_interp2d.py <br>
4.if you want use trajectory shift location info => ~/home/rvl122/paper/dataset/mydataset/trajectory_shift.py <br>

input image info lane line,seg,satellite <br>
1.lane line and seg <br>
training lane line and seg <br>
~/home/rvl122/paper/BiSeNet-ooooverflow/train.py <br>
change (line495)~(line501) <br>

lane line <br>
step1 : open ~/home/rvl122/paper/BiSeNet-ooooverflow/lane_demo.py <br>
step2 : change input and output => (line225),(line232) <br>

seg <br>
step1 : open ~/home/rvl122/paper/BiSeNet-ooooverflow/demo.py <br>
step2 : change input and output => (line261),(line270) <br>

maybe you will use resort  <br>
/home/rvl122/paper/BiSeNet-ooooverflow/resort.py <br>