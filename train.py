from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np


batch_size = 12
gpu = True
lr_ = 5e-4
weight_decay = 5e-4
gamma_ = 0.1

if torch.cuda.is_available():
    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not gpu:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def train():

    cfg = voc
    dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ssd_net.to(device)

    vgg_weights = torch.load('weights/vgg16.pth')
    ssd_net.vgg.load_state_dict(vgg_weights)

    if gpu:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

    # 損失関数の設定
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, gpu)
    # 最適化パラメータの設定
    optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=0.9, weight_decay=weight_decay)

    # 訓練モード
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // batch_size
    print('dataset_size', len(dataset))
    print('epoch_size', epoch_size)
    print('Training SSD on:', dataset.name)

    step_index = 0

    data_loader = data.DataLoader(dataset, batch_size,
                                  num_workers=4,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # 学習の開始
    batch_iterator = None
    # iterationでループして、cfg['max_iter']まで学習する
    for iteration in range(0, 1000):
        # 学習開始時または1epoch終了後にdata_loaderから訓練データをロードする
        if (not batch_iterator) or (iteration % epoch_size ==0):
            batch_iterator = iter(data_loader)
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, gamma_, step_index)

        # load train data
        # バッチサイズ分の訓練データをload
        images, targets = next(batch_iterator)

        # 画像をGPUに転送
        images = images.to(device)
        # アノテーションをGPUに転送
        targets = [ann.to(device) for ann in targets]

        # forward
        t0 = time.time()
        # 順伝播の計算
        out = net(images)
        # 勾配の初期化
        optimizer.zero_grad()
        # 損失関数の計算
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        # 勾配の計算
        loss.backward()
        # パラメータの更新
        optimizer.step()
        t1 = time.time()
        # 損失関数の更新
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        #ログの出力
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

    # 学習済みモデルの保存
    torch.save(ssd_net.state_dict(), 'weights/ssd.pth')


def adjust_learning_rate(optimizer, gamma, step):
    lr = lr_ * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
