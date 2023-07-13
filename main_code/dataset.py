import os
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

#------------------------------------------------------------------------------------------ 

# class MyDataset(Dataset):
#     def __init__(self, path_A, path_B, path_C, path_GPS, path_GT, transform=None):
#         self.path_A = path_A
#         self.path_B = path_B
#         self.path_C = path_C
#         self.path_GPS = path_GPS
#         self.path_GT = path_GT
#         self.transform = transform

#         # self.list_A = os.listdir(self.path_A)
#         # self.list_B = os.listdir(self.path_B)
#         # self.list_C = os.listdir(self.path_C)

#         self.list_A = sorted(glob(os.path.join(self.path_A, "*.jpg")))
#         self.list_B = sorted(glob(os.path.join(self.path_B, "*.png")))
#         self.list_C = sorted(glob(os.path.join(self.path_C, "*.jpg")))

#         with open(self.path_GPS, "r") as f:
#             self.list_GPS = f.readlines()

#         with open(self.path_GT, "r") as f:
#             self.list_GT = f.readlines()


#     def __len__(self):
#         return len(self.list_A)

#     def __getitem__(self, idx):
#         print(self.list_A[idx])
#         print(self.list_B[idx])
#         print(self.list_C[idx])
#         image_A = Image.open(self.list_A[idx])
#         image_B = Image.open(self.list_B[idx])
#         image_C = Image.open(self.list_C[idx])

#         GPS_str  = self.list_GPS[idx].strip().split(",")
#         GPS = torch.tensor([float(GPS_str[0]), float(GPS_str[1])])

#         GT_str = self.list_GT[idx].strip().split(",")
#         GT = torch.tensor([float(GT_str[0]), float(GT_str[1])])

#         if self.transform:
#             image_A = self.transform(image_A)
#             image_B = self.transform(image_B)
#             image_C = self.transform(image_C)

#         return image_A, image_B, image_C, GPS, GT
    
#------------------------------------------------------------------------------------------ 

class MyDataset(Dataset):
    # def __init__(self, path_front, path_seg, path_satellite, path_GPS, path_GT, transform=None):
    def __init__(self, path_front, path_seg, path_satellite, path_line, path_GPS, path_GT, transform=None):
        self.path_A = path_front
        self.path_B = path_seg
        self.path_C = path_satellite
        self.path_D = path_line
        self.path_GPS = path_GPS
        self.path_GT = path_GT
        self.transform = transform

        self.list_A = sorted(glob(os.path.join(self.path_A, "*.jpg")))
        self.list_B = sorted(glob(os.path.join(self.path_B, "*.jpg")))
        self.list_C = sorted(glob(os.path.join(self.path_C, "*.png")))
        self.list_D = sorted(glob(os.path.join(self.path_D, "*.jpg")))

        with open(self.path_GPS, "r") as f:
            self.list_GPS = f.readlines()

        with open(self.path_GT, "r") as f:
            self.list_GT = f.readlines()

    def __len__(self):
        return len(self.list_A)

    def __getitem__(self, idx):
        image_A = Image.open(self.list_A[idx])
        image_B = Image.open(self.list_B[idx])
        image_C = Image.open(self.list_C[idx]).convert("RGB")
        image_D = Image.open(self.list_D[idx])

        GPS_str  = self.list_GPS[idx].strip().split(",")
        GPS = torch.tensor([float(GPS_str[0]), float(GPS_str[1])])

        GT_str = self.list_GT[idx].strip().split(",")
        GT = torch.tensor([float(GT_str[0]), float(GT_str[1])])

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            image_C = self.transform(image_C)
            image_D = self.transform(image_D)
        else:
            # Convert PIL image to PyTorch tensor
            image_A = transforms.ToTensor()(image_A)
            image_B = transforms.ToTensor()(image_B)
            image_C = transforms.ToTensor()(image_C)
            image_D = transforms.ToTensor()(image_D)
        return image_A, image_B, image_C, image_D, GPS, GT
        # return image_A, image_B, image_C, GPS, GT


# if __name__ == "__main__":
#     path_A = "/home/rvl122/paper/dataset/mydataset/Output/image"
#     path_B = "/home/rvl122/paper/Cam2BEV/model/output/2023-03-10-14-39-56/Predictions"
#     path_C = "/home/rvl122/paper/dataset/mydataset/Output/satellite"
#     path_D = "/home/rvl122/paper/dataset/mydataset/Output/line"
#     path_GPS = "/home/rvl122/paper/dataset/mydataset/Output/gpx/Final/interp/latlon_O.txt"
#     path_GT = "/home/rvl122/paper/dataset/mydataset/Output/gpx/Final/interp/latlon.txt"
#     dataset = MyDataset(path_A, path_B, path_C, image_D,path_GPS, path_GT)

#     img_A, img_B, img_C,image_D, GPS, GT = dataset[0]

#     # img_A.show()
#     # img_B.show()
#     # img_C.show()
#     # img_D.show()
#     print(GPS[0], GPS[1])
#     print(GT, GT.shape)

#     print(str(GPS[0].item() - GT[0].item()))