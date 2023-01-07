import itertools
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


class CapchaDataset(Dataset):
    """
    Датасет генерирует капчу длины seq_len из набора данных EMNIST
    """

    def __init__(
        self,
        seq_len: Union[int, Tuple[int, int]],
        img_h: int = 28,
        samples: int = None,
    ):
        digits_dataset = datasets.EMNIST(
            "./EMNIST", split="digits", train=True, download=True
        )
        letters_dataset = datasets.EMNIST(
            "./EMNIST", split="letters", train=True, download=True
        )
        letters_dataset.targets += len(digits_dataset.classes)
        self.datasets = [digits_dataset, letters_dataset]
        self.seq_len = seq_len
        self.blank_label = 10  # 0 blank in letters_dataset
        self.img_h = img_h
        self.samples = samples
        self.num_classes = sum(len(dataset.classes) for dataset in self.datasets)
        self.classes = list(
            itertools.chain([dataset.classes for dataset in self.datasets])
        )
        self.classes = [item for sublist in self.classes for item in sublist]
        self.classes[self.blank_label] = " "  # replace N/A with space
        if isinstance(seq_len, int):
            self._min_seq_len = seq_len
            self._max_seq_len = seq_len
        elif (
            isinstance(seq_len, Tuple)
            and len(seq_len) == 2
            and isinstance(seq_len[0], int)
        ):
            self._min_seq_len = seq_len[0]
            self._max_seq_len = seq_len[1]

    def __len__(self):
        """
        Можно нагенерировать N различных капчей, где N - число сочетаний с повторениями.
        Если задано samples - вернуть его
        """
        if self.samples is not None:
            return self.samples
        return self.num_classes**self._max_seq_len

    def __preprocess(self, random_images: list[torch.Tensor]) -> np.ndarray:
        transformed_images = []
        for img in random_images:
            img = transforms.ToPILImage()(img)
            img = tf.rotate(img, -90, fill=[0.0])
            img = tf.hflip(img)
            img = transforms.ToTensor()(img).numpy()
            transformed_images.append(img)
        images = np.array(transformed_images)
        images = np.hstack(
            images.reshape((len(transformed_images), self.img_h, self.img_h))
        )
        full_img = np.zeros(shape=(self.img_h, self._max_seq_len * self.img_h)).astype(
            np.float32
        )
        full_img[:, 0 : images.shape[1]] = images
        return full_img

    def __getitem__(self, idx):
        # Get random seq_len
        random_seq_len = np.random.randint(self._min_seq_len, self._max_seq_len + 1)
        # Get random ind for dataset like [0, 1, 0, 0, 1]
        random_dataset_indices = np.random.randint(
            len(self.datasets), size=(random_seq_len,)
        )
        random_img_indices = [
            np.random.randint(len(self.datasets[idx].data))
            for idx in random_dataset_indices
        ]
        random_indices = list(zip(random_dataset_indices, random_img_indices))

        random_images = [
            torch.Tensor(self.datasets[idx[0]].data[idx[1]]) for idx in random_indices
        ]
        random_labels = [
            self.datasets[idx[0]].targets[idx[1]] for idx in random_indices
        ]

        labels = torch.zeros((1, self._max_seq_len))
        labels = torch.fill(labels, self.blank_label)
        labels[0, 0 : len(random_labels)] = torch.FloatTensor(random_labels)
        x = self.__preprocess(random_images)
        y = labels.numpy().reshape(self._max_seq_len)
        return x, y


if __name__ == "__main__":
    # от 3 до 5 символов
    ds = CapchaDataset((3, 5))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=2)
    for (x_batch, y_batch) in tqdm(data_loader):
        for img, label in zip(x_batch, y_batch):
            plt.imshow(img)
            title = [str(int(n)) for n in label.numpy()]
            plt.title(" ".join(title))
            plt.show()
