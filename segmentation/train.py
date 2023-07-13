import argparse
import cv2
from ctypes import sizeof
from turtle import shape
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import glob, random
from pprint import pprint
from loss import DiceLoss
import os
import numpy as np
import tqdm
import segmentation_models_pytorch as smp
#from imgaug import augmenters as iaa
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import autocast as autocast
from torchvision import transforms
import torch.profiler as profiler
from torchmetrics.functional.classification import multiclass_jaccard_index 

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from dataset.Dataset import Dataset_preprocess
# from dataset.Dataset_autoalbument import Dataset_preprocess
# from dataset.Dataset_mosaic import Dataset_preprocess
from model.build_BiSeNet import BiSeNet
from model.build_BiseNetv2_tranfer_learning import BiSeNetV2
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu, get_label_info, colour_code_segmentation, get_triangular_lr,  cal_miou
from model.ERFNet.erfnet import ERFNet, Encoder, load_pretrained_encoder

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def val(args, model, dataloader, loss_func, val_step, writer, epoch, csv_path): 
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        # print(model.aux_mode)
        
        model.eval()
        loss_record = []
        loss_step_record = []
        # precision_record = []
        each_img_IOU = 0
        
        hist = np.zeros((args.num_classes, args.num_classes))
        tq = tqdm.tqdm(total=len(dataloader) * args.val_batch_size)
        tq.set_description('Val')
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            if args.models == 'BiseNet':
                predict, predict_sup1, predict_sup2 = model(data, aux_mode='train')
                loss1 = loss_func(predict, label)
                loss2 = loss_func(predict_sup1, label)
                loss3 = loss_func(predict_sup2, label)
                loss = loss1 + loss2 + loss3
            
            elif args.models == 'BiseNetv2':
                predict, predict_sup1, predict_sup2, predict_sup3, predict_sup4 = model(data, aux_mode='train')
                loss1 = loss_func(predict, label)
                loss2 = loss_func(predict_sup1, label)
                loss3 = loss_func(predict_sup2, label)
                loss4 = loss_func(predict_sup3, label)
                loss5 = loss_func(predict_sup4, label)

                loss = loss1 + loss2 + loss3 + loss4 + loss5
            
            elif args.models == 'DeepLabv3+' or args.models == 'Unet' or args.models == 'ERFNet':
                predict = model(data)
                loss = loss_func(predict, label)

            # predict = predict.squeeze()
            # label = label.squeeze()
            # each_img_IOU += multiclass_jaccard_index(torch.argmax(predict, 0),  torch.argmax(label, 0), num_classes=args.num_classes, average = None)

            predict = reverse_one_hot(predict)
            label = reverse_one_hot(label)
            predict = predict.cpu()
            label = label.cpu()
            predict = np.array(predict)
            label = np.array(label)

            # compute per pixel accuracy
            # precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label, predict, args.num_classes, ignore=args.val_ignore_class)

            tq.update(args.val_batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            loss_step_record.append(loss.item())
            loss_record.append(loss.item())
            # precision_record.append(precision)

            if val_step % 50 == 0:
                writer.add_scalar('loss_step/val_step_loss', np.mean(loss_step_record), val_step)
                loss_step_record = []

            val_step += 1

        tq.close()
        loss_val_mean = np.mean(loss_record)

        # csv_path = os.path.join(args.data, args.dataset + '_class_dict.csv')

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

        # each_img_IOU = each_img_IOU.cpu().numpy()
        # miou_dict_jaccard_total, miou_jaccard_total = cal_miou(each_img_IOU, csv_path)
        # miou_jaccard = miou_jaccard_total / (args.val_data_num / 1)
        # # print(each_img_IOU)
        # print('IoU by jaccard for each class:')
        # print('-------------------------------')
        # for key in miou_dict_jaccard_total:
        #     each_iou = miou_dict_jaccard_total[key] / (args.val_data_num / args.val_batch_size)
        #     print('{:30}| {:5.1f} %'.format(key, each_iou * 100))
        # print()
        # print('mIoU for validation: {:.1f} %'.format(miou_jaccard * 100) )

        # writer.add_scalar('eval/precision_val', precision, epoch)
        writer.add_scalar('eval/miou val', miou, epoch)
        # writer.add_scalar('eval/miou_val_jaccard', miou_jaccard, epoch)
        writer.add_scalar('loss/val_loss', float(loss_val_mean), epoch)

        return miou


def predict_on_image(model, args, epoch, writer, csv_path):
    for mode in ('train', 'val'):
        image_list = glob.glob(os.path.join(args.data, mode, '*.*'))
        print(len(image_list))
        image_path = random.choice(image_list)
        label_path = glob.glob(os.path.join(args.data, (mode + '_labels'), str((os.path.split(image_path)[-1])[:-4]) + '*.*'))[0]

        print(mode + ' image path: ', image_path)
        image = cv2.imread(image_path, -1)
        image = cv2.resize(image, (1280, 704))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # print(mode + ' label path: ', label_path)
        label = cv2.imread(label_path, -1)
        label = cv2.resize(label, (1280, 704))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        writer.add_image(mode + '/image', transforms.ToTensor()(np.uint8(image)), epoch)
        writer.add_image(mode + '/label', transforms.ToTensor()(np.uint8(label)), epoch)

        image = Image.fromarray(image).convert('RGB')
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)

        # csv_path = os.path.join(args.data, args.dataset + '_class_dict.csv')
        label_info = get_label_info(csv_path)
        model.eval()

        if args.models == 'BiseNet':
            predict = model(image, aux_mode = 'eval')
        elif args.models == 'BiseNetv2':
            predict = model(image, aux_mode = 'eval')
        elif args.models == 'DeepLabv3+' or args.models == 'Unet' or args.models == 'ERFNet':
            predict = model(image)

        predict = reverse_one_hot(predict.squeeze())
        predict = predict.cpu()
        predict = colour_code_segmentation(np.array(predict), label_info)
        predict = np.uint8(predict)

        if not os.path.isdir(os.path.join(args.save_model_path, mode + '_img')):
            os.makedirs(os.path.join(args.save_model_path, mode + '_img'))
        
        img_save_path = Path(args.save_model_path) / str(mode + '_img') / ( str(epoch) + '.jpg')

        cv2.imwrite(str(img_save_path), cv2.cvtColor(predict, cv2.COLOR_RGB2BGR))

        print('img save to: ', img_save_path)   
        writer.add_image(mode + '/predict_img', transforms.ToTensor()(predict), epoch) 


def train(args, model, optimizer, dataloader_train, dataloader_val, csv_path):
    writer = SummaryWriter(log_dir = os.path.join('runs', str(os.path.split(args.save_model_path)[-1])) ,comment=''.format(args.optimizer, args.context_path))
    '''  
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join('runs', str(os.path.split(args.save_model_path)[-1]))),
        record_shapes=True,
        with_stack=True)
    '''
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        weights = [1.0 for _ in range(args.num_classes)]
        if args.dataset == 'BDD100k' or args.dataset == 'Apollo' or args.dataset == 'Ceymo':
            weights[-1] = 0.05
        elif args.dataset == 'RVL_Dataset':
            weights[0] = 0.2
            # weights = [0.01, 0.27, 0.12, 0.37, 0.17, 0.25, 0.53, 0.32, 0.51, 0.2, 1.61, 0.78, 0.59, 3.11, 0.69, 0.54, 0.28, 0.24, 1.11, 0.77, 0.25, 0.49, 1.84, 2.06, 0.32]

        print('loss weights: {}'.format(weights))
        class_weights = torch.FloatTensor(weights).cuda()
        loss_func = torch.nn.CrossEntropyLoss(weight = class_weights)

    max_miou = 0
    lr = args.learning_rate
    step = 0
    val_step = 0
    scaler = torch.cuda.amp.GradScaler()

    lr_trend = list()
    # precision_trend = list()
    # miou_trend = list()
    # loss_trend = list()
    # max_lr = args.learning_rate

    ## triangular lr range test
    # lr = 0

    for epoch in range(args.epoch_start_i + 1, args.num_epochs + 1):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs, power = 0.9)
        # optimizer.param_groups[0]['lr'] = lr

        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        # tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        loss_step_record = []
        # prof.start()
        for iter, (data, label) in enumerate(dataloader_train):
            # lr = get_triangular_lr(optimizer, iteration = step, stepsize = len(dataloader_train) * 8, base_lr = 0.01, max_lr = 0.12)
            lr_trend.append(lr)
            writer.add_scalar('learning_rate/lr_step', lr, step)
            
            optimizer.zero_grad()
            tq.set_description('epoch %d, lr %f' % (epoch, lr))
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            with autocast():
                if args.models == 'BiseNet':
                    output, output_sup1, output_sup2 = model(data, aux_mode='train')
                    loss1 = loss_func(output, label)
                    loss2 = loss_func(output_sup1, label)
                    loss3 = loss_func(output_sup2, label)
                    loss = loss1 + loss2 + loss3
                elif args.models == 'BiseNetv2':
                    # print(data.size()) 
                    output, output_sup1, output_sup2, output_sup3, output_sup4 = model(data, aux_mode='train')
                    # print(output.size(), label.size())
                    loss1 = loss_func(output, label)
                    loss2 = loss_func(output_sup1, label)
                    loss3 = loss_func(output_sup2, label)
                    loss4 = loss_func(output_sup3, label)
                    loss5 = loss_func(output_sup4, label)
                    loss = loss1 + loss2 + loss3 + loss4 + loss5
                
                elif args.models == 'DeepLabv3+' or args.models == 'Unet' or args.models == 'ERFNet':
                    output = model(data)
                    loss = loss_func(output, label)
                
            
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            loss_step_record.append(loss.item())
            loss_record.append(loss.item())

            if step % (len(dataloader_train)/4) == 0:
                writer.add_scalar('loss_step/train_step_loss', np.mean(loss_step_record), step)
                loss_step_record = []
            

            step += 1
            # prof.step()
        # prof.stop()

        plt.plot(lr_trend)
        plt.title('triangular learning rate')
        plt.savefig(os.path.join(args.save_model_path, 'lr_trend.png'))

        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('loss/train_loss', float(loss_train_mean), epoch)
        writer.add_scalar('learning_rate/lr', lr, epoch)
        print('loss for train : %f' % (loss_train_mean))

        torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest_{}.pth'.format(args.loss)))

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'epoch_{}_{}.pth'.format(epoch, args.loss)))

        if epoch % args.validation_step == 0:
            miou = val(args, model, dataloader_val, loss_func, val_step, writer, epoch, csv_path)

            if miou > max_miou:
                max_miou = miou
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best_{}.pth'.format(args.loss)))

        print('Start visual_val')
        if epoch % args.validation_step == 0:
            predict_on_image(model, args, epoch, writer, csv_path)

        # lr_trend.append(lr)
        # miou_trend.append(miou)
        # precision_trend.append(precision)
        # loss_trend.append(loss.item())

        # print('lr: {} ,miou: {}, precision: {}, loss: {}'.format(lr, miou, precision, loss))
        # plt.plot(lr_trend, miou_trend)
        # plt.title('triangular lr range test_miou')
        # plt.savefig(os.path.join(args.save_model_path, 'lr_range_test_miou.png'))
        # plt.close()

        # plt.plot(lr_trend, precision_trend)
        # plt.title('triangular lr range test_precision')
        # plt.savefig(os.path.join(args.save_model_path, 'lr_range_test_precision.png'))
        # plt.close()

        # plt.plot(lr_trend, loss_trend)
        # plt.title('triangular lr range test_loss')
        # plt.savefig(os.path.join(args.save_model_path, 'lr_range_test_loss.png'))
        # plt.close()

        # lr += 0.01
    

            
def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type = int, default = 300, help = 'Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type = int, default = 0, help = 'Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type = int, default = 1, help = 'How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type = int, default = 1, help = 'How often to perform validation (epochs)')
    parser.add_argument('--dataset', type = str, default = "CamVid", help = 'Dataset you are using.')
    parser.add_argument('--train_data_num', type = int, default = 7000, help = 'Number of training data')
    parser.add_argument('--val_data_num', type = int, default = 1000, help = 'Number of validation data')
    parser.add_argument('--crop_height', type = int, default = 720, help = 'Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type = int, default = 960, help = 'Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'Number of images in each batch for training')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'Number of images in each batch for val')
    parser.add_argument('--context_path', type = str, default = "resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--models', type = str, default = 'BiSeNetv2', help = 'choose which model you want to use, BiseNet or BiseNetv2')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'learning rate used for train')
    parser.add_argument('--data', type = str, default = '', help = 'path of training data')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'num of workers')
    parser.add_argument('--num_classes', type = int, default = 32, help = 'num of object classes (with void)')
    parser.add_argument('--cuda', type = str, default = '0', help = 'GPU ids used for training')
    parser.add_argument('--use_gpu', type = bool, default = True, help = 'whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type = str, default = None, help = 'path to pretrained model')
    parser.add_argument('--save_model_path', type = str, default = None, help = 'path to save model')
    parser.add_argument('--optimizer', type = str, default = 'rmsprop', help = 'optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type = str, default = 'dice', help = 'loss function, dice or crossentropy')
    parser.add_argument('--val_ignore_class', type=int, default=None, help='class number want to ignored' )

    args = parser.parse_args(params)
    for key, value in vars(args).items():
        if value != None:
            print('{:20} = {:<30}'.format(key, value))
    
    args.data = Path(args.data)
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = Path(args.pretrained_model_path)
    args.save_model_path = Path(args.save_model_path)

    ## get image & label folder path 
    train_img_path = os.path.join(args.data, 'train')
    train_label_path = os.path.join(args.data, 'train_labels')
    val_img_path = os.path.join(args.data, 'val')
    val_label_path = os.path.join(args.data, 'val_labels')
     
    csv_path = os.path.join(args.data, args.dataset + '_class_dict.csv')
    # create dataset and dataloader
    dataset_train = Dataset_preprocess(train_img_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                               loss=args.loss, mode='train', data_num = args.train_data_num,  save_model_path = args.save_model_path)
    dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )
    dataset_val = Dataset_preprocess(val_img_path, val_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                             loss=args.loss, mode='val', data_num = args.val_data_num, save_model_path = args.save_model_path)
    dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.val_batch_size, # this has to be 1, otherwise the evaluation will be wrong
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if args.models == 'BiseNet':
        model = BiSeNet(args.num_classes, args.context_path)
    elif args.models == 'BiseNetv2':
        model = BiSeNetV2(args.num_classes)
        # print(model)
    elif args.models == 'DeepLabv3+':
        model = smp.DeepLabV3Plus(classes=args.num_classes)
    elif args.models == 'Unet':
        model = smp.Unet(classes=args.num_classes)
    elif args.models == 'ERFNet':
        model = ERFNet(num_classes=args.num_classes, encoder=load_pretrained_encoder("model/ERFNet/erfnet_encoder_pretrained.pth.tar", 1000))

    setup_seed(20) # setup random seed

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val, csv_path)

    # val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    params = [
        # '--use_gpu', 'False',
        ########################################  ApolloScape  ########################################
        '--models', 'DeepLabv3+', # BiseNet, BiseNetv2, DeepLabv3+, Unet, ERFNet
        '--learning_rate', '0.05', # 0.6 for batch size 16 , 0.3 for batch size 8
        '--context_path', 'resnet18',
        # '--learning_rate', '0.3', # 0.6 for batch size 16 , 0.3 for batch size 8
        '--loss', 'crossentropy',
        '--checkpoint_step', '1',
        '--validation_step', '1',
        '--crop_height','480',
        '--crop_width','800',
        '--num_workers', '8',
        '--cuda', '0',
        '--batch_size', '8',  # 6 for resnet101, 12 for resnet18
        '--val_batch_size', '8',
        '--optimizer', 'sgd',
        ##########################################  BDD_seg_10k  ##########################################
        '--num_epochs', '500',
        '--dataset', 'BDD_seg_10k',
        '--data', 'dataset/BDD_seg_10k',
        '--num_classes', '20',
        '--train_data_num', '7000',
        '--val_data_num', '1000',
        '--save_model_path', 'checkpoints/checkpoints_BDD_seg_10k/BDD_seg_10k_DeepLabv3+_black',
        # '--val_ignore_class', '19',
        # '--pretrained_model_path', 'checkpoints/checkpoints_BDD_seg_10k/BDD_seg_10k_DeepLabv3+/epoch_400_crossentropy.pth',
        # '--epoch_start_i','1',

    ]
    main(params)

