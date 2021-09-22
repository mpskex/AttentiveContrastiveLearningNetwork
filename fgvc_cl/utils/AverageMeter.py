import torch
import shutil


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, ind = scores.topk(k, 1, True, True)
        correct = ind.eq(targets.view(-1, 1).expand_as(ind))
        correct_total = correct.view(-1).float().sum()  # 0D tensor
        return correct_total * (100.0 / batch_size)