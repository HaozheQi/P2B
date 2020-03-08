from torch.utils.data import Dataset
from data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import kitty_utils as utils
from kitty_utils import getModel
from searchspace import KalmanFiltering
import logging


class kittiDataset():

    def __init__(self, path):
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")

    def getSceneID(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                sceneID = [18]
            else:
                sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                sceneID = [19]
            else:
                sceneID = list(range(19, 21))

        else:  # Full Dataset
            sceneID = list(range(21))
        return sceneID

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box

    def getListOfAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        # print(self.list_of_scene)
        list_of_tracklet_anno = []
        for scene in list_of_scene:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == category_name]
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '{:06}.bin'.format(box["frame"]))
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib)
        except :
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data


class SiameseDataset(Dataset):

    def __init__(self,
                 input_size,
                 path,
                 split,
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0):

        self.dataset = kittiDataset(path=path)

        self.input_size = input_size
        self.split = split
        self.sceneID = self.dataset.getSceneID(split=split)
        self.getBBandPC = self.dataset.getBBandPC

        self.category_name = category_name
        self.regress = regress

        self.list_of_tracklet_anno = self.dataset.getListOfAnno(
            self.sceneID, category_name)
        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)


class SiameseTrain(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 sigma_Gaussian=1,
                 offset_BB=0,
                 scale_BB=1.0):
        super(SiameseTrain,self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB)

        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

        self.num_candidates_perframe = 4

        logging.info("preloading PC...")
        self.list_of_PCs = [None] * len(self.list_of_anno)
        self.list_of_BBs = [None] * len(self.list_of_anno)
        for index in tqdm(range(len(self.list_of_anno))):
            anno = self.list_of_anno[index]
            PC, box = self.getBBandPC(anno)
            new_PC = utils.cropPC(PC, box, offset=10)

            self.list_of_PCs[index] = new_PC
            self.list_of_BBs[index] = box
        logging.info("PC preloaded!")

        logging.info("preloading Model..")
        self.model_PC = [None] * len(self.list_of_tracklet_anno)
        for i in tqdm(range(len(self.list_of_tracklet_anno))):
            list_of_anno = self.list_of_tracklet_anno[i]
            PCs = []
            BBs = []
            cnt = 0
            for anno in list_of_anno:
                this_PC, this_BB = self.getBBandPC(anno)
                PCs.append(this_PC)
                BBs.append(this_BB)
                anno["model_idx"] = i
                anno["relative_idx"] = cnt
                cnt += 1

            self.model_PC[i] = getModel(
                PCs, BBs, offset=self.offset_BB, scale=self.scale_BB)

        logging.info("Model preloaded!")

    def __getitem__(self, index):
        return self.getitem(index)

    def getPCandBBfromIndex(self, anno_idx):
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        return this_PC, this_BB

    def getitem(self, index):
        anno_idx = self.getAnnotationIndex(index)
        sample_idx = self.getSearchSpaceIndex(index)

        if sample_idx == 0:
            sample_offsets = np.zeros(3)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 5])
            sample_offsets = gaussian.sample(1)[0]

        this_anno = self.list_of_anno[anno_idx]

        this_PC, this_BB = self.getPCandBBfromIndex(anno_idx)
        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)

        # sample_PC = utils.cropAndCenterPC(
        #     this_PC, sample_BB, offset=self.offset_BB, scale=self.scale_BB)
        sample_PC, sample_label, sample_reg = utils.cropAndCenterPC_label(
            this_PC,sample_BB, this_BB, sample_offsets, offset=self.offset_BB, scale=self.scale_BB)
        if sample_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))
        # sample_PC = utils.regularizePC(sample_PC, self.input_size)[0]
        sample_PC, sample_label, sample_reg = utils.regularizePCwithlabel(sample_PC,sample_label,sample_reg,self.input_size)

        if this_anno["relative_idx"] == 0:
            prev_idx = 0
            fir_idx = 0
        else:
            prev_idx = anno_idx - 1
            fir_idx = anno_idx - this_anno["relative_idx"]
        gt_PC_pre, gt_BB_pre = self.getPCandBBfromIndex(prev_idx)
        gt_PC_fir, gt_BB_fir = self.getPCandBBfromIndex(fir_idx)

        if sample_idx == 0:
            samplegt_offsets = np.zeros(3)
        else:
            samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
            samplegt_offsets[2] = samplegt_offsets[2]*5.0
        gt_BB_pre = utils.getOffsetBB(gt_BB_pre, samplegt_offsets)

        gt_PC = getModel([gt_PC_fir,gt_PC_pre], [gt_BB_fir,gt_BB_pre], offset=self.offset_BB, scale=self.scale_BB)

        if gt_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))
        gt_PC = utils.regularizePC(gt_PC,self.input_size)

        return sample_PC, sample_label, sample_reg, gt_PC

    def __len__(self):
        nb_anno = len(self.list_of_anno)
        return nb_anno * self.num_candidates_perframe

    def getAnnotationIndex(self, index):
        return int(index / (self.num_candidates_perframe))

    def getSearchSpaceIndex(self, index):
        return int(index % self.num_candidates_perframe)


class SiameseTest(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0):
        super(SiameseTest,self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB)
        self.split = split
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        BBs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.getBBandPC(anno)
            PCs.append(this_PC)
            BBs.append(this_BB)
        return PCs, BBs, list_of_anno

    def __len__(self):
        return len(self.list_of_tracklet_anno)


if __name__ == '__main__':

     dataset_Training = SiameseTrain(
            input_size=2048,
            path='/data/qihaozhe/Kitty_data/training',
            split='Tiny_Train',
            category_name='Car',
            offset_BB=0,
            scale_BB=1.15)
     aa = dataset_Training.getitem(201)
     aa = dataset_Training.getitem(30)
     aa = dataset_Training.getitem(100)
     aa = dataset_Training.getitem(120)
     aa = dataset_Training.getitem(200)
     asdf = aa[2].numpy()