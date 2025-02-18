import numpy as np
import h5py
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List
import torch
import cv2


trans_eval = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224))
    ]
)

trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Resize(size=(224,224))
    ]
)

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(
        face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP
    )

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(
            face_model, landmarks, camera, distortion, rvec, tvec, True
        )

    return rvec, tvec

def normalize(
    image, camera_matrix, camera_distortion, face_model_load, landmarks, img_dim
    ):
        landmarks = np.asarray(landmarks)

        # load face model
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = face_model_load[landmark_use, :]
        facePts = face_model.reshape(6, 1, 3)
        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
        landmarks_sub = landmarks_sub.astype(
            float
        )  # input to solvePnP function must be float type
        landmarks_sub = landmarks_sub.reshape(
            6, 1, 2
        )  # input to solvePnP requires such shape
        hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

        # data normalization method
        img_normalized, landmarks_normalized = normalizeData_face(
            image, face_model, landmarks_sub, hr, ht, camera_matrix, img_dim
        )

        return img_normalized

def normalizeData_face(img, face_model, landmarks, hr, ht, cam, img_dim):
            ## normalized camera parameters
        focal_norm = 960  # focal length of normalized camera
        distance_norm = 600  # normalized distance between eye and camera
        roiSize = (img_dim, img_dim)  # size of cropped eye image

        ## compute estimated 3D positions of the landmarks
        ht = ht.reshape((3, 1))
        hR = cv2.Rodrigues(hr)[0]  # rotation matrix
        Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        # get the face center
        face_center = np.mean(
            np.concatenate((two_eye_center, nose_center), axis=1), axis=1
        ).reshape((3, 1))

        ## ---------- normalize image ----------
        distance = np.linalg.norm(
            face_center
        )  # actual distance between eye and original camera

        z_scale = distance_norm / distance
        cam_norm = np.array(
            [  # camera intrinsic parameters of the virtual camera
                [focal_norm, 0, roiSize[0] / 2],
                [0, focal_norm, roiSize[1] / 2],
                [0, 0, 1.0],
            ]
        )
        S = np.array(
            [  # scaling matrix
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, z_scale],
            ]
        )

        hRx = hR[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R

        W = np.dot(
            np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))
        )  # transformation matrix

        img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

        # head pose after normalization
        hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        # normalize the facial landmarks
        num_point = landmarks.shape[0]
        landmarks_warped = cv2.perspectiveTransform(landmarks, W)
        landmarks_warped = landmarks_warped.reshape(num_point, 2)

        return img_warped, landmarks_warped


def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                           transform=trans, is_shuffle=is_shuffle, is_load_label=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

        self.cam_matrix = []
        self.cam_distortion = []
        
        for cam_id in range(18):
            cam_file_name = "cam/cam" + str(cam_id).zfill(2) + ".xml"
            fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
            self.cam_matrix.append(fs.getNode("Camera_Matrix").mat())
            self.cam_distortion.append(fs.getNode("Distortion_Coefficients").mat())
            fs.release()

        self.face_model_load = np.loadtxt("face_model.txt")

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        self.hdf_nerf = h5py.File(
            os.path.join("/data/data2/aruzzi/xgaze_subjects", "xgaze_" + self.selected_keys[key]),
            "r",
            swmr=True,
        )

        # Get face image
        image = self.hdf_nerf['face_patch'][idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        face_mask = self.hdf_nerf["head_mask"][idx, :]
        ldms  = self.hdf_nerf["facial_landmarks"][idx, :]
        cam_ind = self.hdf_nerf["cam_index"][idx, :]

        kernel_2 = np.ones((3, 3), dtype=np.uint8)
        face_mask = cv2.erode(face_mask, kernel_2, iterations=2)
        face_mask = torch.from_numpy(face_mask)
        nonhead_mask = face_mask < 0.5
        nonhead_mask_c3b = nonhead_mask.expand(3, -1, -1)
        image[nonhead_mask_c3b] = 1.0
        """
        image = normalize(
                (image.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8),
                self.cam_matrix[int(cam_ind)],
                self.cam_distortion[int(cam_ind)],
                self.face_model_load,
                ldms,
                224,
        )
        """
        image = trans_eval(image)
        
        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            #gaze_label = self.hdf_nerf['pitchyaw_head'][idx, :]
            gaze_label = gaze_label.astype('float')
            head_label = self.hdf['face_head_pose'][idx, :]
            #head_label = self.hdf_nerf['face_head_pose'][idx, :]
            head_label = head_label.astype('float')
            return image, gaze_label, head_label
        else:
            return image