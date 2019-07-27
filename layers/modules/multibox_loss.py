import torch
import torch.nn as nn
import torch.nn.functional as F
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        # 推論結果をオフセット、確信度、ボックス座標にセット
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        # 正解座標のオフセット、正解ラベルのテンソルを作成
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # バッチサイズ毎にループし、訓練データを正解座標、正解ラベルに分解
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # 正解座標とボックス座標のマッチング
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            loc_t = loc_t.to(device)
            conf_t = conf_t.to(device)
        # wrap targets

        # クラス番号が0より大きいPositiveのボックスのリスト作成
        pos = conf_t > 0
        # Positiveのボックス数
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # Positiveのボックスのインデックスpos_idxを取得
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 推論結果のオフセット
        loc_p = loc_data[pos_idx].view(-1, 4)
        # 正解座標のオフセット
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 位置の損失関数
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 推論結果の確信度conf_dataをpos_idx+neg_idxで絞り込み
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # 正解ラベルのconf_tをposとnegで絞り込み
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # クラス確信度の損失関数
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)


        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
