import torch
import os
import copy
import numpy as np
from pyquaternion import Quaternion
from data_classes import PointCloud
from metrics import estimateOverlap
from scipy.optimize import leastsq


def distanceBB_Gaussian(box1, box2, sigma=1):
    off1 = np.array([
        box1.center[0], box1.center[2],
        Quaternion(matrix=box1.rotation_matrix).degrees
    ])
    off2 = np.array([
        box2.center[0], box2.center[2],
        Quaternion(matrix=box2.rotation_matrix).degrees
    ])
    dist = np.linalg.norm(off1 - off2)
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return score


# IoU or Gaussian score map
def getScoreGaussian(offset, sigma=1):
    coeffs = [1, 1, 1 / 5]
    dist = np.linalg.norm(np.multiply(offset, coeffs))
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return torch.tensor([score])


def getScoreIoU(a, b):
    score = estimateOverlap(a, b)
    return torch.tensor([score])


def getScoreHingeIoU(a, b):
    score = estimateOverlap(a, b)
    if score < 0.5:
        score = 0.0
    return torch.tensor([score])


def getOffsetBB(box, offset):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    # REMOVE TRANSfORM
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[3] * np.pi / 180))
    if offset[0]>new_box.wlh[0]:
        offset[0] = np.random.uniform(-1,1)
    if offset[1]>min(new_box.wlh[1],2):
        offset[1] = np.random.uniform(-1,1)
    new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def voxelize(PC, dim_size=[48, 108, 48]):
    # PC = normalizePC(PC)
    if np.isscalar(dim_size):
        dim_size = [dim_size] * 3
    dim_size = np.atleast_2d(dim_size).T
    PC = (PC + 0.5) * dim_size
    # truncate to integers
    xyz = PC.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dim_size), 0)
    xyz = xyz[:, valid_ix]
    out = np.zeros(dim_size.flatten(), dtype=np.float32)
    out[tuple(xyz)] = 1
    # print(out)
    return out


def regularizePC2(input_size, PC,):
    return regularizePC(PC=PC, input_size=input_size)


def regularizePC(PC,input_size):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != int(input_size/2):
            np.random.seed(1)
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=int(input_size/2), dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape((3, int(input_size/2))).T

    else:
        PC = np.zeros((3, int(input_size/2))).T

    return torch.from_numpy(PC).float()

def regularizePCwithlabel(PC,label,reg, input_size):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != input_size:
            np.random.seed(1)
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
            label = label[new_pts_idx]
        PC = PC.reshape((3, input_size)).T
        label = label[0:128]
        reg = np.tile(reg,[np.size(label),1])

    else:
        PC = np.zeros((3, input_size)).T
        label = np.zeros(128)
        reg = np.tile(reg,[np.size(label),1])

    return torch.from_numpy(PC).float(),torch.from_numpy(label).float(),torch.from_numpy(reg).float()


def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):

    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))

    for PC, box in zip(PCs, boxes):
        cropped_PC = cropAndCenterPC(
            PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points, cropped_PC.points], axis=1)

    PC = PointCloud(points)

    return PC


def cropPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC

def getlabelPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate((rot_mat))
    box_tmp.rotate(Quaternion(matrix=(rot_mat)))
    
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_label = np.zeros(new_PC.points.shape[1])
    new_label[close] = 1
    return new_label

def cropPCwithlabel(PC, box,label, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    new_label = label[close]
    return new_PC,new_label

def weight_process(include,low,high):
    if include<low:
        weight = 0.7
    elif include >high:
        weight = 1
    else:
        weight = (include*2.0+3.0*high-5.0*low)/(5*(high-low))
    return weight

def func(a, x):
    k, b = a
    return k * x + b
def dist(a, x, y):
    return func(a, x) - y

def weight_process2(k):
    k = abs(k)
    if k>1:
        weight = 0.7
    else:
        weight = 1-0.3*k
    return weight



def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC

def Centerbox(sample_box, gt_box):
    rot_mat = np.transpose(gt_box.rotation_matrix)
    trans = -gt_box.center

    new_box = copy.deepcopy(sample_box)
    new_box.translate(trans)
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    return new_box


def cropAndCenterPC_label(PC, sample_box, gt_box,sample_offsets, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
    new_box_gt = copy.deepcopy(gt_box)
    # new_box_gt2 = copy.deepcopy(gt_box)

    #rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+2.0, scale=1 * scale)
    #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)

    label_reg = [new_box_gt.center[0],new_box_gt.center[1],new_box_gt.center[2],-sample_offsets[2]]
    label_reg = np.array(label_reg)
    # label_reg = np.tile(label_reg,[np.size(new_label),1])

    if normalize:
        new_PC.normalize(sample_box.wlh)
    return new_PC, new_label, label_reg

def cropAndCenterPC_label_test_time(PC, sample_box, offset=0, scale=1.0):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset+2.0, scale=scale)

    return new_PC

def cropAndCenterPC_label_test(PC, sample_box, gt_box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
    new_box_gt = copy.deepcopy(gt_box)
    # new_box_gt2 = copy.deepcopy(gt_box)

    rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))
    # new_box_gt2.translate(trans)
    # new_box_gt2.rotate(rot_quat.inverse)

    # crop around box
    new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+2.0, scale=1 * scale)
    #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)

    label_reg = [new_box_gt.center[0],new_box_gt.center[1],new_box_gt.center[2]]
    label_reg = np.array(label_reg)
    # label_reg = (new_PC.points - np.tile(new_box_gt.center,[np.size(new_label),1]).T) * np.expand_dims(new_label, axis=0)
    # newoff = [new_box_gt.center[0],new_box_gt.center[1],new_box_gt.center[2]]
    # newoff = np.array(newoff)

    if normalize:
        new_PC.normalize(sample_box.wlh)
    return new_PC, new_label, label_reg, new_box, new_box_gt

def distanceBB(box1, box2):

    eucl = np.linalg.norm(box1.center - box2.center)
    angl = Quaternion.distance(
        Quaternion(matrix=box1.rotation_matrix),
        Quaternion(matrix=box2.rotation_matrix))
    return eucl + angl


def generate_boxes(box, search_space=[[0, 0, 0]]):
    # Geenrate more candidate boxes based on prior and search space
    # Input : Prior position, search space and seaarch size
    # Output : List of boxes

    candidate_boxes = [getOffsetBB(box, offset) for offset in search_space]
    return candidate_boxes


def getDataframeGT(anno):
    df = {
        "scene": anno["scene"],
        "frame": anno["frame"],
        "track_id": anno["track_id"],
        "type": anno["type"],
        "truncated": anno["truncated"],
        "occluded": anno["occluded"],
        "alpha": anno["alpha"],
        "bbox_left": anno["bbox_left"],
        "bbox_top": anno["bbox_top"],
        "bbox_right": anno["bbox_right"],
        "bbox_bottom": anno["bbox_bottom"],
        "height": anno["height"],
        "width": anno["width"],
        "length": anno["length"],
        "x": anno["x"],
        "y": anno["y"],
        "z": anno["z"],
        "rotation_y": anno["rotation_y"]
    }
    return df


def getDataframe(anno, box, score):
    myquat = (box.orientation * Quaternion(axis=[1, 0, 0], radians=-np.pi / 2))
    df = {
        "scene": anno["scene"],
        "frame": anno["frame"],
        "track_id": anno["track_id"],
        "type": anno["type"],
        "truncated": anno["truncated"],
        "occluded": anno["occluded"],
        "alpha": 0.0,
        "bbox_left": 0.0,
        "bbox_top": 0.0,
        "bbox_right": 0.0,
        "bbox_bottom": 0.0,
        "height": box.wlh[2],
        "width": box.wlh[0],
        "length": box.wlh[1],
        "x": box.center[0],
        "y": box.center[1] + box.wlh[2] / 2,
        "z": box.center[2],
        "rotation_y":
        np.sign(myquat.axis[1]) * myquat.radians,  # this_anno["rotation_y"], #
        "score": score
    }
    return df


def saveTrackingResults(df_3D, dataset_loader, export=None, epoch=-1):

    for i_scene, scene in enumerate(df_3D.scene.unique()):
        new_df_3D = df_3D[df_3D["scene"] == scene]
        new_df_3D = new_df_3D.drop(["scene"], axis=1)
        new_df_3D = new_df_3D.sort_values(by=['frame', 'track_id'])

        os.makedirs(os.path.join("results", export, "data"), exist_ok=True)
        if epoch == -1:
            path = os.path.join("results", export, "data", "{}.txt".format(scene))
        else:
            path = os.path.join("results", export, "data",
                                "{}_epoch{}.txt".format(scene,epoch))

        new_df_3D.to_csv(
            path, sep=" ", header=False, index=False, float_format='%.6f')
