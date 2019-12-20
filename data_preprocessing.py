import xlrd
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import copy
import os
import zipfile
from cache import cache

config = [
    dict(source_data='/data/PETScanner/data_new/data.pkl', split=0.8, output_path='/data/PETScanner/dataset_new.pkl'),
]


def normalize(x):
    return x / x.sum()


def main():
    np.random.seed(0)
    for cfg in config:
        print("\nCreating dataset for config:\n\tsplit: {split}\n\tsource_data: {source_data}\n\tsaving to {output_path}".format(**cfg))
        create_dataset(**cfg)


def create_dataset(source_data, split, output_path):
    with open(source_data, 'rb') as f:
        data = pickle.load(f)

    np.random.shuffle(data)

    p = int(len(data) * split)
    data_train = data[:p]
    data_validation = data[p:]

    data_train = [(normalize(img), y) for img, y in data_train]
    data_validation = [(normalize(img), y) for img, y in data_validation]

    with open(output_path, 'wb') as f:
        pickle.dump(dict(train=data_train, validation=data_validation), f)


if __name__ == "__main__":
    main()
