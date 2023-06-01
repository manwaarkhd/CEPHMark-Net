from data.isbi_dataset import ISBIDataset
from data.pku_dataset import PKUDataset
from data.dataset import Dataset
from config import cfg

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    datasets_root_path = "../datasets"
    isbi_dataset_root = os.path.join(datasets_root_path, "ISBI Dataset")

    data = Dataset(name="ISBI", mode="TRAIN", batch_size=4, shuffle=False)

    images, landmarks = data[0]

    index = 0
    plt.imshow(images[index])
    plt.scatter(landmarks[index, :, 0], landmarks[index, :, 1], color="green", s=[3] * cfg.NUM_LANDMARKS)
    plt.show()
