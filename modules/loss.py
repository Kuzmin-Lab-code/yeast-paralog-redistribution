import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5, s=30.0, m=0.5, one_hot_labels=False):
        super().__init__()
        self.alpha = alpha
        self.ce = DenseCrossEntropy() if one_hot_labels else nn.CrossEntropyLoss()
        self.af = ArcFaceLoss(s=s, m=m, one_hot_labels=one_hot_labels)

    def forward(self, logits, labels):
        loss_ce = self.ce(logits, labels)
        loss_af = self.af(logits, labels)
        loss = self.alpha * loss_af + (1 - self.alpha) * loss_ce
        return loss


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5, one_hot_labels=False):
        super().__init__()
        self.one_hot_labels = one_hot_labels
        self.criterion = DenseCrossEntropy()
        #         (
        #     DenseCrossEntropy() if one_hot_labels else nn.CrossEntropyLoss()
        # )
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = labels
        if not self.one_hot_labels:
            one_hot = torch.zeros(cosine.size(), device="cuda")
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        #         output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.criterion(output, one_hot)
        # TODO remove / 2 ?
        return loss / 2


class ArcMarginProductPlain(nn.Module):
    def __init__(self, in_embeddings, out_embeddings):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_embeddings, in_embeddings))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, embeddings):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine
