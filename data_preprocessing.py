import xlrd
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import random
import copy
import os
import zipfile
from cache import cache


config = [
    dict(split=0.8, shared=1.0, output_path='/data/PETScanner/dataset_1.pkl'),
    dict(split=0.8, shared=0.5, output_path='/data/PETScanner/dataset_2.pkl'),
    dict(split=0.8, shared=0.0, output_path='/data/PETScanner/dataset_3.pkl'),
]


def main():
    data, image_list = read_tiff_data()

    emission_points = get_emission_points()

    for cfg in config:
        print("\nCreating dataset for config:\n\tsplit: {split}\n\tshared: {shared}\n\tsaving to {output_path}".format(**cfg))
        create_dataset(data, emission_points, image_list, **cfg)


def create_dataset(data, emission_points, image_list, shared, split, output_path):
    position_list = get_position_list(image_list)
    emissions_shuffled = copy.deepcopy(position_list)
    random.shuffle(emissions_shuffled)
    random.shuffle(image_list)
    positions_shared, positions_train, position_validation = split_positions(emissions_shuffled, shared, split)
    train_images, validation_images = split_images(image_list, position_list, position_validation, positions_shared,
                                                   positions_train, split)
    validate_split(train_images, validation_images, shared, split)
    data_train = [(normalize(data[image]), get_ground_truth(image, position_list, emission_points)) for image in train_images]
    data_validation = [(normalize(data[image]), get_ground_truth(image, position_list, emission_points)) for image in
                       validation_images]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dict(train=data_train, validation=data_validation), f)


@cache
def read_tiff_data():
    archive = get_archive()
    image_list = [x.filename for x in archive.filelist if x.filename.endswith('.tiff')]
    data = read_images(archive, image_list)
    return data, image_list


def validate_split(train_images, validation_images, shared, split):
    print("Number of train images: %d" % len(train_images))
    print("Number of validation images: %d" % len(validation_images))
    ratio = (len(train_images) * 1.0 / (len(train_images) + len(validation_images)))
    print("Split: %f" % ratio)
    assert abs(ratio - split) < 0.01, "Wrong split"
    train_set = set(train_images)
    shared_images = len([x for x in validation_images if x in train_set])
    assert shared_images == 0, "Train and validation set must be disjoint!"

    position_list_train = set([extract_position_from_filename(x) for x in train_images])
    position_list_validation = set([extract_position_from_filename(x) for x in validation_images])
    positions_shared = position_list_train.intersection(position_list_validation)

    number_of_shared = len([x for x in validation_images if extract_position_from_filename(x) in positions_shared])
    ratio_of_shared = number_of_shared * 1.0 / len(validation_images)
    print("Ratio of shared: %f" % ratio_of_shared)
    assert abs(ratio_of_shared - shared) < 0.01, "Wrong amount of shared"
    print('Validation passed with shared: {shared}, split: {split}'.format(shared=shared, split=split))


def split_images(image_list, position_list, position_validation, positions_shared, positions_train, split):
    images = {x: [] for x in position_list}
    for image in image_list:
        images[extract_position_from_filename(image)].append(image)
    train_images = sum(
        [images[x][:int(len(images[x]) * split)] for x in positions_shared] + [images[x] for x in positions_train], [])
    validation_images = sum(
        [images[x][int(len(images[x]) * split):] for x in positions_shared] + [images[x] for x in position_validation], [])
    return set(train_images), set(validation_images)


def split_positions(emissions_shuffled, shared, split):
    number_of_shared_positions = int(len(emissions_shuffled) * shared)
    number_of_non_shared_positions = len(emissions_shuffled) - number_of_shared_positions
    positions_shared = emissions_shuffled[:number_of_shared_positions]
    positions_non_shared = emissions_shuffled[number_of_shared_positions:]
    positions_train = positions_non_shared[:int(number_of_non_shared_positions * split)]
    position_validation = positions_non_shared[int(number_of_non_shared_positions * split):]
    return set(positions_shared), set(positions_train), set(position_validation)


def normalize(x):
    return x / x.sum()


def read_images(archive, image_list):
    return {img: np.asarray(Image.open(archive.open(img))) for img in tqdm(image_list)}


def get_archive():
    return zipfile.ZipFile("/data/PETScanner/Data/tiffs.zip", 'r')


def extract_position_from_filename(img):
    img = img.split('.')
    img = img[0].split('/')
    img = img[-1].split('_')
    return int(img[1])


def get_position_list(image_list):
    position_list = [extract_position_from_filename(x) for x in image_list]
    position_list = list(set(position_list))
    position_list.sort()
    return position_list


def get_ground_truth(image, position_list, emission_points):
    p = extract_position_from_filename(image)
    i = position_list.index(p)
    return np.asarray(emission_points[i])


def get_emission_points():
    location = '/data/PETScanner/Data/emission_coordinates.xlsx'
    wb = xlrd.open_workbook(location)
    sheet = wb.sheet_by_index(0)
    emission_points = []
    # position_list has all the positions in the tiffs with index number+1 as the row in the excel sheet
    for i in range(1, 822):
        emission_points.append(sheet.row_values(i - 1))
    return emission_points


if __name__ == "__main__":
    main()
