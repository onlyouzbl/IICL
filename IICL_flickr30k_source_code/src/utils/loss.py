from torch import nn
import torch
import numpy as np
from torch.nn.functional import normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cosine_dist(im, s):
    """Cosine similarity between all the image and sentence pairs
    """

    return 1 - im.mm(s.t())


def euclidean_dist(x, y):
    """
      Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
      Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist


class TripletLoss(nn.Module):
    """Triplet loss class

    Parameters
    ----------
    margin : float
        Ranking loss margin
    metric : string
        Distance metric (either euclidean or cosine)
    """

    def __init__(self, margin=0.3, metric='cosine'):

        super(TripletLoss, self).__init__()
        self.distance_function = euclidean_dist if metric == 'euclidean' else cosine_dist
        self.metric = metric
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, im, s):
        # compute image-sentence score matrix
        # batch_size x batch_size
        '''
            q-im: torch.Size([256,512])
            v-s: torch.Size([256,512])
        '''
        scores_i2r = self.distance_function(normalize(im, dim=-1),
                                            normalize(s, dim=-1))    #(256,256)
        scores_r2i = scores_i2r.t()

        pos = torch.eye(im.size(0))
        neg = 1 - pos

        pos = (pos == 1).to(im.device)
        neg = (neg == 1).to(im.device)

        # positive similarities
        # batch_size x 1
        # python diag()函数功能：1. 二维数组取出对角线上的元素   2.一维数组将数组中的每个元素作为对角线上元素形成二维数组
        d1 = scores_i2r.diag().view(im.size(0), 1)     #(256,1)
        d2 = d1.t()     #转置 (1,256)

        y = torch.ones(scores_i2r.size(0)).to(im.device)

        # image anchor - recipe positive bs x bs
        #  expand_as（）函数与expand（）函数类似，功能都是用来扩展张量中某维数据的尺寸，区别是它括号内的输入参数是另一个张量，
        #  作用是将输入tensor的维度扩展为与指定tensor相同的size
        d1 = d1.expand_as(scores_i2r)
        # recipe anchor - image positive
        d2 = d2.expand_as(scores_i2r) #bs x bs

        y = y.expand_as(scores_i2r)

        # compare every diagonal score to scores in its column
        # recipe retrieval
        # batch_size x batch_size (each anchor is compared to all elements in the batch)
        cost_im = self.ranking_loss(scores_i2r, d1, y)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_s = self.ranking_loss(scores_i2r, d2, y)

        # clear diagonals
        cost_s = cost_s.masked_fill_(pos, 0)
        cost_im = cost_im.masked_fill_(pos, 0)

        return (cost_s + cost_im).mean()

class MILNCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MILNCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):
        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]
        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator
