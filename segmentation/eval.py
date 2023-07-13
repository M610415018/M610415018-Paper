import torch
import argparse
import os
import numpy as np
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
# from model.build_BiseNetv2 import BiSeNetV2
from model.build_BiseNetv2_tranfer_learning import BiSeNetV2
# from dataset.CamVid import CamVid
# from dataset.BDD100k import BDD100k
# from dataset.Apollo import Apollo
from dataset.Dataset import Dataset_preprocess
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou, get_label_info
from torchmetrics.functional.classification import multiclass_jaccard_index 
import segmentation_models_pytorch as smp
from model.ERFNet.erfnet import ERFNet, Encoder, load_pretrained_encoder


def eval(model,dataloader, args, csv_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        # tq = tqdm.tqdm(total=len(dataloader))
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        # each_img_IOU = 0
        # read csv label path
        label_info = get_label_info(args.csv_path)
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            # tq.update(1)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            
            if args.models == 'BiseNet':
                predict = model(data, aux_mode = 'eval')
            elif args.models == 'BiseNetv2':
                # predict = torch.as_tensor(model(data)[0]).squeeze()
                predict = (model(data, aux_mode='eval'))
            elif args.models == 'DeepLabv3+' or args.models == 'Unet' or args.models == 'ERFNet':
                predict = model(data)
            # IOU_tensor = multiclass_jaccard_index(predict.flatten(),  label.flatten(), num_classes=args.num_classes)
            # each_img_IOU += multiclass_jaccard_index(torch.argmax(predict, 0),  torch.argmax(label.squeeze(), 0), num_classes=args.num_classes, average = None)
            # predict = model(data).squeeze()
            # predict = torch.as_tensor(model(data)[0]).squeeze()
            
            predict = reverse_one_hot(predict)
            label = reverse_one_hot(label)
            predict = predict.cpu()
            # label = label.squeeze()
            label = label.cpu()
            predict = np.array(predict)
            label = np.array(label)

            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)

            # precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label, predict, args.num_classes, ignore=args.val_ignore_class)
            # precision_record.append(precision)
            
        tq.close()
        # precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        print('miou_list:', miou_list)
        miou_dict, miou = cal_miou(miou_list, csv_path)
        print('IoU for each class:')
        print('-------------------------------')
        for key in miou_dict:
            # print('{}:{:.3f},'.format(key, miou_dict[key]))
            each_iou = miou_dict[key]
            print('{:30}| {:5.1f} %'.format(key, each_iou * 100))

        # print('precision for test: %.3f' % precision)
        # print('mIoU for validation: %.3f' % miou)

        # print('precision for validation: {:.1f} %'.format(precision * 100) )
        print('mIoU for validation: {:.1f} %'.format(miou * 100) )

        # print()

        # each_img_IOU = each_img_IOU.cpu()
        # each_img_IOU = each_img_IOU.numpy()
        # miou_dict, miou = cal_miou(each_img_IOU, csv_path)
        # print(each_img_IOU)
        # print()
        # print('IoU for each class:')
        # print('-------------------------------')
        # for key in miou_dict:
        #     each_IOU = miou_dict[key] / (args.val_data_num / args.batch_size)
        #     print('{:22}| {:5.1f} %'.format(key, each_IOU * 100) )
        # print()
        # miou_percentage = (miou / (args.val_data_num / args.batch_size))*100
        # print('mIoU for validation: {:.1f} %'.format(miou_percentage) )

        return None

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, required=False, help='Dataset you are using.')
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='The path to the class csv')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--data_path', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.') 
    parser.add_argument('--models', type = str, default = 'BiSeNetv2', help = 'choose which model you want to use, BiseNet or BiseNetv2')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--val_data_num', type=int, default = 1000, help='Number of validation data')
    parser.add_argument('--val_ignore_class', type=int, default=None, help='class number want to ignored' )
    args = parser.parse_args(params)

    for key, value in vars(args).items():
        if value != None:
            print('{:20} = {:<30}'.format(key, value))

    # # create dataset and dataloader

    ## get image & label folder path 
    val_img_path = os.path.join(args.data_path, 'val')
    val_label_path = os.path.join(args.data_path, 'val_labels')
     
    # csv_path = os.path.join(args.data_path, args.dataset + '_class_dict.csv')
    csv_path = args.csv_path
    print(csv_path)
    # create dataset and dataloader
    dataset_val = Dataset_preprocess(val_img_path, val_label_path, csv_path, scale=(args.crop_height, args.crop_width),loss=args.loss, 
                            mode='val', data_num = args.val_data_num, dataset=args.dataset)
    dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )
    # csv_path = os.path.join(args.data, 'class_dict.csv')

    # dataloader = DataLoader(
    #     dataset,
    #     # batch_size=args.batch_size,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    # )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if args.models == 'BiseNet':
        model = BiSeNet(args.num_classes, args.context_path)
    elif args.models == 'BiseNetv2':
        model = BiSeNetV2(args.num_classes)
    elif args.models == 'DeepLabv3+':
        model = smp.DeepLabV3Plus(classes=args.num_classes)
    elif args.models == 'Unet':
        model = smp.Unet(classes=args.num_classes)
    elif args.models == 'ERFNet':
        model = ERFNet(num_classes=args.num_classes, encoder=load_pretrained_encoder("model/ERFNet/erfnet_encoder_pretrained.pth.tar", 1000))

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    # model_dict = torch.load(args.checkpoint_path)['state_dict']
    # new_model_dict = dict()
    # for key, value in model_dict.items():
    #     new_key = key[6:]
    #     if key[:5] == 'model':
    #         print(value)
    #         new_model_dict[new_key] = value

    model.module.load_state_dict(torch.load(args.checkpoint_path))
    # model.module.load_state_dict(new_model_dict, strict=True)
    
    print('Done!')

    # get label info
    # label_info = get_label_info(csv_path)
    # test
    eval(model, dataloader_val, args, csv_path)


if __name__ == '__main__': 
    params = [
        # '--models', 'BiseNetv2',
        '--models', 'DeepLabv3+', 
        # '--models', 'Unet',
        # '--models', 'ERFNet',
        # '--dataset', 'RVL_Dataset',
        '--crop_height', '480',
        '--crop_width', '800',
        '--cuda', '0',
        '--batch_size', '2',  # 6 for resnet101, 12 for resnet18
        '--num_workers', '8',
        '--context_path', 'resnet18',  # only support resnet18 and resnet101
        ##########################################  BDD_seg_10k  ##########################################
        # '--checkpoint_path', 'checkpoints/checkpoints_BDD_seg_10k/BDD_seg_10k_DeepLabv3+/best_crossentropy.pth',
        '--checkpoint_path', 'checkpoints/checkpoints_BDD_seg_10k/BDD_seg_10k_DeepLabv3+_black/epoch_800_crossentropy.pth',
        '--csv_path', 'dataset/BDD_seg_10k/BDD_seg_10k_class_dict.csv',
        '--data_path', 'dataset/BDD_seg_10k',
        '--val_data_num', '1000',
        '--num_classes', '20',
        '--val_ignore_class', '20',

    ]

    main(params)