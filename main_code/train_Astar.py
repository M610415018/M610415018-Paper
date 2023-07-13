import os
import time, datetime
import pathlib
from tqdm import tqdm, trange
from model import FC_Model
from dataset import MyDataset
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import heapq

# Add A* search function
def A_star(start, end, image_line, GPS, model, device):
    """
    A* search function to find the shortest path
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while len(frontier) > 0:
        current = heapq.heappop(frontier)[1]
        if current == end:
            break
        
        for next in neighbors(current, image_line):
            next_tensor = torch.tensor(next, dtype=torch.float32, device=device).unsqueeze(0)
            current_tensor = torch.tensor(current, dtype=torch.float32, device=device).unsqueeze(0)
            new_cost = cost_so_far[current] + model_distance(GPS.to(device) + next_tensor, GPS.to(device) + current_tensor, model)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(end, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return came_from

def heuristic(a, b):
    """
    Simple Euclidean distance heuristic
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def neighbors(current, image_line):
    """
    Return a list of valid neighbors
    """
    (i, j) = current
    candidates = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
    result = []
    for (x, y) in candidates:
        if 0 <= x < image_line.shape[0] and 0 <= y < image_line.shape[1] and image_line[x, y] == 0:
            result.append((x, y))
    return result

def model_distance(a, b, model):
    """
    Calculate the distance between two GPS points using the trained model
    """
    delta = model(a, b)
    return np.sqrt(delta[0]**2 + delta[1]**2)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def fit_one_epoch(net, criterion, epoch, epoch_size, Epoch, cuda):
    total_loss = 0
    save_path = ("/home/rvl122/paper/main/checkpoints/junyi/")
    # Train the model and save the weights
    with tqdm(total=epoch_size, desc='Epoch {}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
        for i, (images_front, images_seg, images_satellite, image_line, GPS, target) in enumerate(dataloader):
            if cuda:
                images_front = images_front.cuda()
                images_seg = images_seg.cuda()
                images_satellite = images_satellite.cuda()
                image_line = image_line.cuda()
                GPS = GPS.cuda()
                target = target.cuda()
            model.new_output = model.new_output.detach() / 1000
            start = (int(GPS[0][0].item()), int(GPS[0][1].item()))
            end = (int(target[0][0].item()), int(target[0][1].item()))
            came_from = A_star(start, end, image_line.squeeze().cpu().numpy(), GPS[0].cpu().numpy(), model, device)
            path = [end]
            current = end
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()

            # Convert path from list to tensor
            path = np.array(path)
            path = torch.from_numpy(path).to(device)
            path = torch.stack((path[:, 0], path[:, 1]), dim=1)
            output = model(images_front, images_seg, images_satellite, image_line)   # shape of output = (B, 2); (delta_x, delta_y)

            # Add GPS data to update new_output
            model.new_output = (GPS.to(device) + output)*1000
            # Compute the A* loss
            astar_loss = 0
            for j in range(path.shape[0] - 1):
                point1 = path[j]
                point2 = path[j+1]
                distance = model_distance(point1.unsqueeze(0), point2.unsqueeze(0), model)
                astar_loss += distance

            # Combine the A* loss with the existing loss
            loss_lon = criterion(model.new_output[0][0], (target[0][0]*1000).to(device))
            loss_lat = criterion(model.new_output[0][1], (target[0][1]*1000).to(device))
            loss = (loss_lon + (loss_lat*1.2)) + astar_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(**{'total_loss': total_loss / (i + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
            # Log the loss to Tensorboard
            writer.add_scalar('Training/Loss', loss.item(), epoch * len(dataloader) + i)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(save_path, 'model_epoch_{}.pth'.format(epoch+1)))

if __name__ == '__main__':

    # path_front = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/images"
    # path_seg = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/seg"
    # path_line = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/line"
    # path_satellite = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/satellite"
    # path_GPS = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/gpx/shift_test.txt"
    # path_GT = "/home/rvl122/paper/dataset/mydataset/ITRI_Zhongxing/ITRI_Zhongxing_01/gpx/init/ITRI_Zhongxing_01_GT.txt"


    # path_front = "/home/rvl122/paper/dataset/mydataset/HIGHWAYS/images"
    # path_seg = "/home/rvl122/paper/dataset/mydataset/HIGHWAYS/seg"
    # path_line = "/home/rvl122/paper/dataset/mydataset/HIGHWAYS/line"
    # path_satellite = "/home/rvl122/paper/dataset/mydataset/HIGHWAYS/satellite"
    # path_GPS = "/home/rvl122/paper/dataset/mydataset/HIGHWAYS/gpx/init/highways_fake.txt"
    # path_GT = "/home/rvl122/paper/dataset/mydataset/HIGHWAYS/gpx/init/highways_GT.txt"


    # path_front = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_002/image"
    # path_seg = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_002/seg"
    # path_line = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_002/line"
    # path_satellite = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_002/satellite"
    # path_GPS = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_002/gpx/shift_test3-5_01_o.txt"
    # path_GT = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_002/gpx/0002_GT_lonlat.txt"

    # path_front = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/image"
    # path_seg = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/seg"
    # path_line = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/line"
    # path_satellite = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/satellite"
    # path_GPS = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx/shift_test3-5_01.txt"
    # path_GT = "/home/rvl122/paper/dataset/mydataset/KITTI_dataset/test_005/gpx/0005_GT_lonlat.txt"

    # path_front = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/image"
    # path_seg = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/seg"
    # path_line = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/line"
    # path_satellite = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/satellite_512"
    # path_GPS = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/shift/shift_test3-5.txt"
    # path_GT = "/home/rvl122/paper/dataset/mydataset/CCU/14322_all_dataset/gpx/Final/interp/init/GT_latlon.txt"

    path_front = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/Image"
    path_seg = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/seg"
    path_line = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/line"
    path_satellite = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/satellite"
    path_GPS = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gps.txt"
    path_GT = "/home/rvl122/paper/dataset/mydataset/junyi_dataset/220531153403/gpx/220531153403_gt.txt"


# Define hyperparameters
    cuda = True
    pre_train = True
    CosineLR = False
    lr = 1e-3
    batch_size = 1
    Init_Epoch = 0
    Fin_Epoch = 100

    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Define the datasets and dataloaders
    dataset = MyDataset(path_front, path_seg, path_satellite ,path_line, path_GPS, path_GT,transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FC_Model()
    model.to(device)
    # if pre_train:
    #     model.load_state_dict(torch.load("/home/rvl122/paper/main/checkpoints/RTK/Japan_1_2.pth", map_location=device)["model_state_dict"])
    model.train()
    epoch_size = len(dataloader)
    # Create a SummaryWriter object for logging
    writer = SummaryWriter()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if CosineLR:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-10)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(Init_Epoch, Fin_Epoch):
        fit_one_epoch(net=model, criterion=criterion, epoch=epoch, epoch_size=epoch_size,Epoch=Fin_Epoch, cuda=cuda)
        lr_scheduler.step()

    writer.close()
    print('Finished Training')