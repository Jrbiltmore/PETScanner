import xlrd
import numpy as np
import glob
from PIL import Image
import pickle
import tqdm
import random
import copy
import os


def main():
    location = 'Data/emission_coordinates.xlsx'

    wb = xlrd.open_workbook(location)
    sheet = wb.sheet_by_index(0)
    emission_points = []
    for _ in range(1, 822):
        emission_points.append(sheet.row_values(_))
    #print("emission points before shuffling: ", emission_points)
    random.shuffle(emission_points)
    #print("emission points after shuffling: ", emission_points)
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


    emissions_shuffled = copy.deepcopy(position_list)
    random.shuffle(emissions_shuffled)
    validation1 = emissions_shuffled[:400]
    random.shuffle(emissions_shuffled)
    validation2 = emissions_shuffled[:300]
    random.shuffle(emissions_shuffled)
    validation3 = emissions_shuffled[:200]

    input1_X = []
    output1_Y = []
    input2_X = []
    output2_Y = []
    input3_X = []
    output3_Y = []
    validation1_input_X = []
    validation1_output_Y = []
    validation2_input_X = []
    validation2_output_Y = []
    validation3_input_X = []
    validation3_output_Y = []
    images_shuffles = [i for i in range(1, 200)]
    random.shuffle(images_shuffles)
    for img in tqdm.tqdm(glob.glob('Data/tiffs/*.tiff')):
        image = np.asarray(Image.open(img))


        img = img.split('.')
        img = img[0].split('/')
        img = img[-1].split('_')
        i = position_list.index(int(img[1]))
        j = int(img[3])
        if i in validation1 and j in images_shuffles[0:101]:

            validation1_input_X.append(image)
            validation1_output_Y.append(np.asarray([
            emission_points[i][0],
            emission_points[i][1],
            emission_points[i][2]]))
        else:
            input1_X.append(image)
            output1_Y.append(np.asarray([
                emission_points[i][0],
                emission_points[i][1],
                emission_points[i][2]]))
    for img in tqdm.tqdm(glob.glob('Data/tiffs/*.tiff')):
        image = np.asarray(Image.open(img))


        img = img.split('.')
        img = img[0].split('/')
        img = img[-1].split('_')
        i = position_list.index(int(img[1]))
        j = int(img[3])
        if i in validation2[0:100] or (i in emissions_shuffled[101:] and j in images_shuffles[0:101]):

            validation2_input_X.append(image)
            validation2_output_Y.append(np.asarray([
            emission_points[i][0],
            emission_points[i][1],
            emission_points[i][2]]))
        else:
            input2_X.append(image)
            output2_Y.append(np.asarray([
                emission_points[i][0],
                emission_points[i][1],
                emission_points[i][2]]))
    for img in tqdm.tqdm(glob.glob('Data/tiffs/*.tiff')):
        image = np.asarray(Image.open(img))


        img = img.split('.')
        img = img[0].split('/')
        img = img[-1].split('_')
        i = position_list.index(int(img[1]))
        j = int(img[3])
        if i in validation3:

            validation3_input_X.append(image)
            validation3_output_Y.append(np.asarray([
            emission_points[i][0],
            emission_points[i][1],
            emission_points[i][2]]))
        else:
            input3_X.append(image)
            output3_Y.append(np.asarray([
                emission_points[i][0],
                emission_points[i][1],
                emission_points[i][2]]))

    # print(len(output_Y))
    #
    # print("input vectors length", len(input_X))
    #
    # train_set_size = int(0.8 * len(input_X))
    #
    # all_data = [(x, y) for x, y in zip(input_X, output_Y)]

    np.random.seed(1)

    #random.shuffle(all_data)

    #input_X = [x[0] for x in all_data]
    #output_Y = [x[1] for x in all_data]

    #train_X = input_X[:train_set_size]
    #train_Y = output_Y[:train_set_size]

    #validation_X = input_X[train_set_size:]
    #validation_Y = output_Y[train_set_size:]

    # input1_X = []
    # output1_Y = []
    # input2_X = []
    # output2_Y = []
    # input3_X = []
    # output3_Y = []
    # validation1_input_X = []
    # validation1_output_Y = []
    # validation2_input_X = []
    # validation2_output_Y = []
    # validation3_input_X = []
    # validation3_output_Y = []
    plotsave = 'New_Dataset/Validation_1/'
    try:
        os.makedirs(plotsave)
    except OSError:
        print("Directory already exists")
    else:
        print("Directory " + plotsave + " is created successfully")

    with open(plotsave + 'train1_X.pkl', 'wb') as f:
        pickle.dump(input1_X, f)

    with open(plotsave + 'train1_Y.pkl', 'wb') as f:
        pickle.dump(output1_Y, f)

    with open(plotsave + 'validation1_X.pkl', 'wb') as f:
        pickle.dump(validation1_input_X, f)

    with open(plotsave + 'validation1_Y.pkl', 'wb') as f:
        pickle.dump(validation1_output_Y, f)

    plotsave = 'New_Dataset/Validation_2/'
    try:
        os.mkdir(plotsave)
    except OSError:
        print("Directory already exists")
    else:
        print("Directory " + plotsave + " is created successfully")

    with open(plotsave + 'train2_X.pkl', 'wb') as f:
        pickle.dump(input2_X, f)

    with open(plotsave + 'train2_Y.pkl', 'wb') as f:
        pickle.dump(output2_Y, f)

    with open(plotsave + 'validation2_X.pkl', 'wb') as f:
        pickle.dump(validation2_input_X, f)

    with open(plotsave + 'validation2_Y.pkl', 'wb') as f:
        pickle.dump(validation2_output_Y, f)

    plotsave = 'New_Dataset/Validation_3/'
    try:
        os.mkdir(plotsave)
    except OSError:
        print("Directory already exists")
    else:
        print("Directory " + plotsave + " is created successfully")

    with open(plotsave + 'train3_X.pkl', 'wb') as f:
        pickle.dump(input3_X, f)

    with open(plotsave + 'train3_Y.pkl', 'wb') as f:
        pickle.dump(output3_Y, f)

    with open(plotsave + 'validation3_X.pkl', 'wb') as f:
        pickle.dump(validation3_input_X, f)

    with open(plotsave + 'validation3_Y.pkl', 'wb') as f:
        pickle.dump(validation3_output_Y, f)




if __name__ == "__main__":
    main()
