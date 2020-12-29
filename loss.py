# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from: https://github.com/fartashf/vsepp/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
    YmX = YmX - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, hard=False, device="cuda:0"):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == "order":
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.hard = hard
        self.device = device

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        mask_i = Variable(mask)
        if torch.cuda.is_available():
            mask_i = mask_i.to(self.device)
        cost_s = cost_s.masked_fill_(mask_i, 0)
        cost_im = cost_im.masked_fill_(mask_i, 0)

        # keep the maximum violating negative for each query
        if self.hard:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
