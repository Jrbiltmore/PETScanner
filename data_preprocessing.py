import xlrd
import numpy as np
import glob
from PIL import Image
import torch
import pickle
import tqdm


def main():
    location = 'Data/emission_coordinates.xlsx'

    wb = xlrd.open_workbook(location)
    sheet = wb.sheet_by_index(0)

    # print(sheet.cell_value(0, 0))
    position_list = []
    count = 0
    for img in glob.glob('Data/tiffs/*.tiff'):
        img = img.split('.')
        img = img[0].split('/')
        img = img[-1].split('_')
        position_list.append(int(img[1]))

        count += 1
    position_list = list(set(position_list))
    position_list.sort()
    # position_list has all the positions in the tiffs with index number+1 as the row in the excel sheet

    input_X = []
    output_Y = []
    for img in tqdm.tqdm(glob.glob('Data/tiffs/*.tiff')):
        image = np.asarray(Image.open(img))

        input_X.append(image)
        img = img.split('.')
        img = img[0].split('/')
        img = img[-1].split('_')
        i = position_list.index(int(img[1]))
        output_Y.append(np.asarray([
            sheet.cell_value(i+1, 0),
            sheet.cell_value(i+1, 1),
            sheet.cell_value(i+1, 2)]))

    print(len(output_Y))

    print("input vectors length", len(input_X))

    train_set_size = int(0.8 * len(input_X))

    train_X = input_X[:train_set_size]
    train_Y = output_Y[:train_set_size]

    validation_X = input_X[train_set_size:]
    validation_Y = output_Y[train_set_size:]

    with open('train_X.pkl', 'wb') as f:
        pickle.dump(train_X, f)

    with open('train_Y.pkl', 'wb') as f:
        pickle.dump(train_Y, f)

    with open('validation_X.pkl', 'wb') as f:
        pickle.dump(validation_X, f)

    with open('validation_Y.pkl', 'wb') as f:
        pickle.dump(validation_Y, f)


if __name__ == "__main__":
    main()
