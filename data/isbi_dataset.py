from config import cfg
import numpy as np
import random
import cv2
import os


class ISBIDataset(object):

    def __init__(
        self,
        dataset_folder_path: str,
        mode: str
    ) -> None:

        if mode in ["train", "valid", "test"]:
            self.mode = mode
        else:
            raise ValueError("mode could only be train, valid or test")

        self.imgaes_root_path = os.path.join(dataset_folder_path, self.mode, "images")

        self.annotations_root_path = os.path.join(dataset_folder_path, self.mode, "annotations")
        self.senior_annotations_root = os.path.join(self.annotations_root_path, "senior-orthodontist")
        self.junior_annotations_root = os.path.join(self.annotations_root_path, "junior-orthodontist")

        self.images_list = list(sorted(os.listdir(self.imgaes_root_path)))

    def __getitem__(self, index: int):
        image_file_name = self.images_list[index]
        label_file_name = self.images_list[index].split(".")[0] + "." + "txt"

        image = self.get_image(image_file_name)
        label = self.get_label(label_file_name)

        return image, label

    def get_image(self, file_name: str):
        file_path = os.path.join(self.imgaes_root_path, file_name)

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def get_label(self, file_name: str) -> np.ndarray:
        file_path = os.path.join(self.senior_annotations_root, file_name)
        with open(file_path) as file:
            senior_annotations = [landmark.rstrip() for landmark in file]

        senior_annotations = [[float(landmark.split(",")[0]), float(landmark.split(",")[1])] for landmark in senior_annotations[:cfg.NUM_LANDMARKS]]
        senior_annotations = np.array(senior_annotations, dtype=np.float32)

        file_path = os.path.join(self.junior_annotations_root, file_name)
        with open(file_path) as file:
            junior_annotations = [landmark.rstrip() for landmark in file]

        junior_annotations = [[float(landmark.split(",")[0]), float(landmark.split(",")[1])] for landmark in junior_annotations[:cfg.NUM_LANDMARKS]]
        junior_annotations = np.array(junior_annotations, dtype=np.float32)

        landmarks = np.zeros(shape=(cfg.NUM_LANDMARKS, 2), dtype=np.int32)
        landmarks[:, 0] = np.ceil((0.5) * (junior_annotations[:, 0] + senior_annotations[:, 0]))
        landmarks[:, 1] = np.ceil((0.5) * (junior_annotations[:, 1] + senior_annotations[:, 1]))

        return np.array(landmarks, dtype=np.float32)

    def shuffle(self):
        random.shuffle(self.images_list)

    def __len__(self):
        return len(self.images_list)
