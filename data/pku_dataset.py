from config import cfg
import numpy as np
import random
import cv2
import os


class PKUDataset(object):

    def __init__(
        self,
        dataset_folder_path: str,
        mode: str = None
    ):
        self.images_root_path = os.path.join(dataset_folder_path, "images")
        self.labels_root_path = os.path.join(dataset_folder_path, "annotations")

        self.doctor1_annotations_root = os.path.join(self.labels_root_path, "orthodontist-1")
        self.doctor2_annotations_root = os.path.join(self.labels_root_path, "orthodontist-2")

        self.images_list = os.listdir(self.images_root_path)

    def __getitem__(self, index):
        image_file_name = self.images_list[index]
        label_file_name = self.images_list[index].split(".")[0] + "." + "txt"

        image = self.get_image(image_file_name)
        landmarks = self.get_label(label_file_name)

        return image, landmarks

    def get_image(self, file_name: str):
        file_path = os.path.join(self.images_root_path, file_name)

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return np.array(image, dtype=np.uint8)

    def get_label(self, file_name: str):
        file_path = os.path.join(self.doctor1_annotations_root, file_name)
        with open(file_path) as file:
            doctor1_annotations = [landmark.rstrip() for landmark in file]

        doctor1_annotations = [[float(landmark.split(",")[0]), float(landmark.split(",")[1])] for landmark in doctor1_annotations[:cfg.NUM_LANDMARKS]]
        doctor1_annotations = np.array(doctor1_annotations, dtype=np.float32)

        file_path = os.path.join(self.doctor2_annotations_root, file_name)
        with open(file_path) as file:
            doctor2_annotations = [landmark.rstrip() for landmark in file]

        doctor2_annotations = [[float(landmark.split(",")[0]), float(landmark.split(",")[1])] for landmark in doctor2_annotations[:cfg.NUM_LANDMARKS]]
        doctor2_annotations = np.array(doctor2_annotations, dtype=np.float32)

        landmarks = np.zeros(shape=(cfg.NUM_LANDMARKS, 2), dtype=np.int32)
        landmarks[:, 0] = np.ceil((0.5) * (doctor1_annotations[:, 0] + doctor2_annotations[:, 0]))
        landmarks[:, 1] = np.ceil((0.5) * (doctor1_annotations[:, 1] + doctor2_annotations[:, 1]))

        return np.array(landmarks, dtype=np.float32)

    def shuffle(self):
        random.shuffle(self.images_list)

    def __len__(self):
        return len(self.images_list)
