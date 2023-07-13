import cv2
import argparse
import os
import torch
import cv2
import tqdm
import numpy as np
import time
from model.build_BiSeNet import BiSeNet
# from model.build_BiseNetv2 import BiSeNetV2
from model.build_BiseNetv2_tranfer_learning import BiSeNetV2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
from natsort import natsorted
from pprint import pprint
import segmentation_models_pytorch as smp


def predict_on_image(model, args, image_path: str, save_path: str):
    # pre-processing on image
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')

    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image_tensor).unsqueeze(0)
    # read csv label path
    label_info = get_label_info(args.csv_path)
    # predict
    model.eval()
    # predict = model(image).squeeze()
    if args.models == 'BiseNet' or args.models == 'BiseNetv2':
        predict = model(image_tensor, aux_mode = 'eval')
    elif args.models == 'DeepLabv3+':
        predict = model(image_tensor)
        
    predict = torch.as_tensor(predict).squeeze()
    predict = reverse_one_hot(predict)
    predict = predict.cpu()
    predict = colour_code_segmentation(np.array(predict), label_info)
    # predict = cv2.resize(np.uint8(predict), (1280, 720))
    predict = cv2.resize(np.uint8(predict), (args.crop_width, args.crop_height))
    predict = cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR)

    # image_overlay = np.where(predict == (0, 0, 0), cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR), predict)
    ################### overlay color w black ####################
    if args.overlay_img:
        predict_black_pixels = np.where(
                (predict[:, :, 0] == 0) &
                (predict[:, :, 1] == 0) &
                (predict[:, :, 2] == 0))
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
        predict[predict_black_pixels] = image[predict_black_pixels]

    cv2.imwrite(save_path, predict)
    # print('img save to: ', save_path)


def predict_on_video(model, args, video_path: str, save_path: str):

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    Output_Video = cv2.VideoWriter(save_path,fourcc, video_fps, (args.crop_width, args.crop_height))  # create empty video
    tq = tqdm.tqdm(total = cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tq.set_description('Video ')
    inf_time_total = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            # print('break')
            # break
            continue

        tq.update(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (args.crop_width, args.crop_height))
        

        resize = iaa.Scale({'width': args.crop_width, 'height': args.crop_height})
        resize_det = resize.to_deterministic()
        image = resize_det.augment_image(image)
        image = Image.fromarray(image).convert('RGB')
        # image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)

        
        # image = transforms.ToTensor()(image)
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image_tensor).unsqueeze(0)
        # read csv label path
        label_info = get_label_info(args.csv_path)
        # predict
        model.eval()
        inf_t1 = time.time()
        # predict = model(image_tensor)[0]
        # predict = model(image_tensor, aux_mode='eval')
        if args.models == 'BiseNet' or args.models == 'BiseNetv2':
            predict = model(image_tensor, aux_mode = 'eval')
        elif args.models == 'DeepLabv3+':
            predict = model(image_tensor)

        inf_time_total += time.time() - inf_t1
        predict = torch.as_tensor(predict).squeeze()
        predict = reverse_one_hot(predict)
        predict = predict.cpu()
        predict = colour_code_segmentation(np.array(predict), label_info)
        predict = cv2.resize(np.uint8(predict), (args.crop_width, args.crop_height))
        predict = cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR)

        # image_overlay = np.where(predict == (0, 0, 0), cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR), predict)
        ################### overlay color w black ####################
        if args.overlay_img:
            predict_black_pixels = np.where(
                (predict[:, :, 0] == 0) &
                (predict[:, :, 1] == 0) &
                (predict[:, :, 2] == 0))
            image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
            predict[predict_black_pixels] = image[predict_black_pixels]

        # image_overlay = np.where(predict == (0, 0, 0), image, predict)
        # print(ret, end = '')

        Output_Video.write(predict)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    print('Total Frame: ', video_total_frame)
    print('Inf_time: ', inf_time_total)
    print('FPS: ', video_total_frame / inf_time_total)
    Output_Video.release()
    cap.release()
    # cv2.destroyAllWindows()

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=bool, default=True, help='predict on image')
    parser.add_argument('--image', type=bool, default=False, help='predict on image')
    parser.add_argument('--video', type=bool, default=False, help='predict on video')
    # parser.add_argument('--image', action='store_true', default=True, help='predict on image')
    # parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--models', type = str, default = 'BiSeNetv2', help = 'choose which model you want to use, BiseNet or BiseNetv2')
    # parser.add_argument('--num_classes', type=int, default=33, help='num of object classes (with void)')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')
    parser.add_argument('--overlay_img', type=bool, default=False, help='Overlay image or not')


    args = parser.parse_args(params)
    
    for key, value in vars(args).items():
        if value != None:
            print('{:20} = {:<30}'.format(key, value))

    # if not os.path.exists(args.save_path.rsplit("/",1)[0]) :
    #     os.makedirs(args.save_path.rsplit("/",1)[0])
    
    if not os.path.exists(args.save_path) :
        os.makedirs(args.save_path)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if args.models == 'BiseNet':
        model = BiSeNet(args.num_classes, args.context_path)
    elif args.models == 'BiseNetv2':
        model = BiSeNetV2(args.num_classes)
    elif args.models == 'DeepLabv3+':
        model = smp.DeepLabV3Plus(classes=args.num_classes)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # predict on image
    if args.image:
        print('Predict on image')
        if os.path.isdir(args.data):
            image_list = natsorted(os.listdir(args.data))
            save_path = os.path.join(args.save_path, 'image_' + args.checkpoint_path.split('/')[-1].split('.')[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # pprint(image_list)
            tq = tqdm.tqdm(total = len(image_list))
            for file in image_list:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = os.path.join(args.data, file)
                    img_save_path = os.path.join(save_path, image_path.split('/')[-1])
                    predict_on_image(model, args, image_path, img_save_path)
                    tq.update(1)
        else:
            image_save_name = args.checkpoint_path.split('/')[-1].split('.')[0] + '_' + args.data.split('/')[-1]
            image_save_path = os.path.join(args.save_path, image_save_name)
            predict_on_image(model, args, args.data, image_save_path)
            print('img save to: ', args.save_path)

    # predict on video
    if args.video:
        print('Predict on video : ')
        if os.path.isdir(args.data):
            video_list = natsorted(os.listdir(args.data))
            save_path = os.path.join(args.save_path, 'video_' + args.checkpoint_path.split('/')[-1].split('.')[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # pprint(video_list)
            for file in video_list:
                print(file)
                if file.endswith('.mkv') or file.endswith('.mp4') or file.endswith('.MOV'):
                    video_path = os.path.join(args.data, file)
                    video_save_path = os.path.join(save_path, video_path.split('/')[-1].split('.')[0] + '.mp4')
                    print('video_path: {}'.format(video_path))
                    print('video_save_path: {}'.format(video_save_path))
                    predict_on_video(model, args, video_path, video_save_path)
        else:
            video_save_name = args.checkpoint_path.split('/')[-1].split('.')[0] + '_' + args.data.split('/')[-1].split('.')[0] + '.mp4'
            video_save_path = os.path.join(args.save_path, video_save_name)
            print(video_save_path)
            predict_on_video(model, args, args.data, video_save_path)

if __name__ == '__main__':

    params = [
        '--models', 'DeepLabv3+',
        # '--models', 'BiseNetv2',
        '--crop_height','1088',
        '--crop_width','1920',
        '--context_path', 'resnet18',
        # '--overlay_img', 'True',
        ########################################  Video  ########################################
        # '--video', 'True',
        # '--data', '/home/rvl122/paper/dataset/mydataset/video/GH014322.MP4', # video folder to inference
        # '--data', 'inference/inputs/videos/main_inference_folder/day_6_Trim.mp4', # chiayi_To_Changhua dataset
        # '--data', 'inference/inputs/videos/Chiayi_To_Changhua/day_1_1.mp4', # Chiayi_To_Changhua dataset
        # '--data', 'inference/inputs/videos/Chiayi_To_Changhua/day_3.mp4', # Chiayi_To_Changhua dataset
        # '--data', 'inference/inputs/videos/20221115/GH014149_1_min.mp4', # Chiayi_To_Tainan dataset
        # '--data', 'inference/inputs/videos/220209144515.MOV', # chiayi_city
        # '--data', 'inference/inputs/videos/220209135331.MOV', # chiayi_city
        # '--data', 'inference/inputs/videos/Beijing_Drive_on_East_3rd_Ring_Road_cut.mp4', # Beijing

        ########################################  Image  ########################################
        '--image', 'True',
        '--data', '/home/rvl122/paper/dataset/mydataset/junyi_dataset/201116145712/Image', # image folder to inference
        # '--data', 'dataset/RVL_Dataset/RVLDataset_w_marker/val', # image folder to inference
        # '--data', 'dataset/RVL_Dataset/RVLDataset_w_marker/val/40.jpg',
        # '--data', 'inference/inputs/images/xiadengren/10.jpg',

        ##########################################  BDD_seg_10k  ##########################################
        '--checkpoint_path', '/home/rvl122/paper/BiSeNet-ooooverflow/checkpoints/checkpoints_BDD_seg_10k/BDD_seg_10k_DeepLabv3+_black/best_crossentropy.pth',
        '--csv_path', 'dataset/BDD_seg_10k/BDD_seg_10k_class_dict.csv',
        '--num_classes', '20',
        '--save_path', '/home/rvl122/paper/dataset/mydataset/junyi_dataset/201116145712/seg',

    ]
    main(params)
