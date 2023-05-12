
import os
import torch
import argparse

from lib.model import CFANet
from utils.dataloader import get_loader,test_dataset
from utils.trainer import adjust_lr
from datetime import datetime

import torch.nn.functional as F
import numpy as np
import logging

best_mae   = 1
best_epoch = 0


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def train(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :return:
    """
    model.train()
    
    
    size_rates = [0.75, 1, 1.25]
    
       
    for step, data_pack in enumerate(train_loader):
        
        images, gts, egs = data_pack
        
        for rate in size_rates:
            
            optimizer.zero_grad()
            
            images = images.cuda()
            gts    = gts.cuda()
            egs    = egs.cuda()
            
            
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts    = F.upsample(gts,    size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                egs    = F.upsample(egs,    size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        

            cam_edge, sal_out1, sal_out2, sal_out3 = model(images)
            loss_edge  = loss_func(cam_edge, egs)
            loss_sal1  = structure_loss(sal_out1, gts)
            loss_sal2  = structure_loss(sal_out2, gts)
            loss_sal3  = structure_loss(sal_out3, gts)
            
            loss_total = loss_edge + loss_sal1 + loss_sal2 + loss_sal3
        
            loss_total.backward()
            optimizer.step()

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_edge: {:.4f} Loss_sal1: {:0.4f} Loss_sal2: {:0.4f} Loss_sal3: {:0.4f} Loss_total: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_edge.data, loss_sal1.data, loss_sal2.data, loss_sal3.data, loss_total.data))
            
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_edge: {:.4f} Loss_sal1: {:0.4f} Loss_sal2: {:0.4f} Loss_sal3: {:0.4f} Loss_total: {:0.4f}'.
                    format( epoch, opt.epoch, step, total_step, loss_edge.data, loss_sal1.data, loss_sal2.data, loss_sal3.data, loss_total.data))


    if (epoch) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'CODNet_%d.pth' % (epoch))
        
        
def test(test_loader,model,epoch,save_path):
    
    global best_mae,best_epoch
    model.eval()
    
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            
            gt /= (gt.max() + 1e-8)
            
            image = image.cuda()

            _,_,_,res = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum +=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
            
        mae = mae_sum / test_loader.size
      
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae   = mae
                best_epoch = epoch
                
                torch.save(model.state_dict(), save_path+'/Cod_best.pth')
                print('best epoch:{}'.format(epoch))
                
       
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int,   default=200,  help='epoch number, default=30')
    parser.add_argument('--lr',          type=float, default=1e-4, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize',   type=int,   default=10,   help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize',   type=int,   default=352,  help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip',        type=float, default=0.5,  help='gradient clipping margin')
    parser.add_argument('--decay_rate',  type=float, default=0.1,  help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int,   default=30,   help='every N epochs decay lr')
    parser.add_argument('--gpu',         type=int,   default=0,    help='choose which gpu you use')
    parser.add_argument('--save_epoch',  type=int,   default=5,    help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model',  type=str,   default='./Snapshot/CFANet/')
    
    
    parser.add_argument('--train_img_dir', type=str, default='./data/TrainDataset/images/')
    parser.add_argument('--train_gt_dir',  type=str, default='./data/TrainDataset/masks/')
    parser.add_argument('--train_eg_dir',  type=str, default='./data/TrainDataset/edges/')
    
    parser.add_argument('--test_img_dir',  type=str, default='./data/TestDataset/CVC-300/images/')
    parser.add_argument('--test_gt_dir',   type=str, default='./data/TestDataset/CVC-300/masks/')
    parser.add_argument('--test_eg_dir',   type=str, default='./data/TestDataset/CVC-300/edges/')
    
    
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)
    
    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)
    
    
    logging.basicConfig(filename=opt.save_model+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("COD-Train")
    logging.info("Config")
    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.save_model,opt.decay_epoch))



    # TIPS: you also can use deeper network for better performance like channel=64
    model = CFANet(channel=64).cuda()
    #print('-' * 30, model, '-' * 30)
    
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter:%.2fM' % (total/1e6))



    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()

    #net, optimizer = amp.initialize(model_SINet, optimizer, opt_level='O1')     # NOTES: Ox not 0x

    train_loader = get_loader(opt.train_img_dir,  opt.train_gt_dir, opt.train_eg_dir, batchsize=opt.batchsize,trainsize=opt.trainsize, num_workers=12)
    test_loader  = test_dataset(opt.test_img_dir, opt.test_gt_dir,  testsize=opt.trainsize)
    
    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)

    for epoch_iter in range(1, opt.epoch):
        
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        
        train(train_loader, model, optimizer, epoch_iter,opt, LogitsBCE, total_step)
        #test(test_loader,   model, epoch_iter, opt.save_model)
        
        
        
