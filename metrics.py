import numpy as np
from shapely.geometry import Polygon


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def estimateAccuracy(box_a, box_b, dim=3):
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        return np.linalg.norm(
            box_a.center[[0, 2]] - box_b.center[[0, 2]], ord=2)


def fromBoxToPoly(box):
    return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))


def estimateOverlap(box_a, box_b, dim=2):
    # if box_a == box_b:
    #     return 1.0

    Poly_anno = fromBoxToPoly(box_a)
    Poly_subm = fromBoxToPoly(box_b)

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area

    else:

        ymax = min(box_a.center[1], box_b.center[1])
        ymin = max(box_a.center[1] - box_a.wlh[2],
                   box_b.center[1] - box_b.wlh[2])

        inter_vol = box_inter.area * max(0, ymax - ymin)
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)

    return overlap


class Success(object):
    """Computes and stores the Success"""

    def __init__(self, n=21, max_overlap=1):
        self.max_overlap = max_overlap
        self.Xaxis = np.linspace(0, self.max_overlap, n)
        self.reset()

    def reset(self):
        self.overlaps = []

    def add_overlap(self, val):
        self.overlaps.append(val)

    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):
        succ = [
            np.sum(i >= thres
                   for i in self.overlaps).astype(float) / self.count
            for thres in self.Xaxis
        ]
        return np.array(succ)

    @property
    def average(self):
        if len(self.overlaps) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_overlap


class Precision(object):
    """Computes and stores the Precision"""

    def __init__(self, n=21, max_accuracy=2):
        self.max_accuracy = max_accuracy
        self.Xaxis = np.linspace(0, self.max_accuracy, n)
        self.reset()

    def reset(self):
        self.accuracies = []

    def add_accuracy(self, val):
        self.accuracies.append(val)

    @property
    def count(self):
        return len(self.accuracies)

    @property
    def value(self):
        prec = [
            np.sum(i <= thres
                   for i in self.accuracies).astype(float) / self.count
            for thres in self.Xaxis
        ]
        return np.array(prec)

    @property
    def average(self):
        if len(self.accuracies) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_accuracy

