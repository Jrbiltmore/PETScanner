import torch
import xlrd



location = 'Data/emission_coordinates.xlsx'

wb = xlrd.open_workbook(location)
sheet = wb.sheet_by_index(0)

#print(sheet.cell_value(0, 0))



position_list = []
import glob
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


from PIL import Image
import numpy as np

count = 0
input_X = []
output_Y = []
for img in glob.glob('Data/tiffs/*.tiff'):

    image_matrix = np.asarray(Image.open(img))

    image_tensor = torch.from_numpy(image_matrix)

    temp = []
    for i in range(len(image_tensor)):
        for j in range(len(image_tensor[0])):
            temp.append(image_tensor[i][j])
    input_X.append(torch.stack(temp))
    img = img.split('.')
    img = img[0].split('/')
    img = img[-1].split('_')
    i = position_list.index(int(img[1]))
    output_Y.append(torch.stack([torch.tensor(sheet.cell_value(i+1,0)), torch.tensor(sheet.cell_value(i+1,1)),
                                 torch.tensor(sheet.cell_value(i+1,2))]))

print(len(output_Y))

import pickle
file = open('input_tensor_data', 'wb')
file1 = open('output_tensor_data', 'wb')
pickle.dump(output_Y, file1)
pickle.dump(input_X, file)
file.close()
file1.close()


print("***** Data in the file input_tensor_data")
file = open('input_tensor_data', 'rb')
data = pickle.load(file)
print("input vectors length",len(data))
file.close()
file1 = open('output_tensor_data', 'rb')
data = pickle.load(file1)
print("output vectors length", len(data))
file1.close()

count = 0
input_X = []
output_Y = []
X_validation, y_validation = [], []
for img in glob.glob('Data/tiffs/*.tiff'):

    image_matrix = np.asarray(Image.open(img))

    image_tensor = torch.from_numpy(image_matrix)

    temp = []
    for i in range(len(image_tensor)):
        for j in range(len(image_tensor[0])):
            temp.append(image_tensor[i][j])

    img = img.split('.')
    img = img[0].split('/')
    img = img[-1].split('_')
    i = position_list.index(int(img[1]))
    if int(i)>=599:
        X_validation.append(torch.stack(temp))
        y_validation.append(torch.stack([torch.tensor(sheet.cell_value(i + 1, 0)), torch.tensor(sheet.cell_value(i + 1, 1)),
                                     torch.tensor(sheet.cell_value(i + 1, 2))]))
    else:
        input_X.append(torch.stack(temp))
        output_Y.append(torch.stack([torch.tensor(sheet.cell_value(i+1,0)), torch.tensor(sheet.cell_value(i+1,1)),
                                     torch.tensor(sheet.cell_value(i+1,2))]))
import pickle
file = open('Train_1_input', 'wb')
file1 = open('Train_1_output', 'wb')
pickle.dump(output_Y, file1)
pickle.dump(input_X, file)
file.close()
file1.close()

file = open('Test_1_validation_input', 'wb')
file1 = open('Test_1_validation_output', 'wb')
pickle.dump(y_validation, file1)
pickle.dump(X_validation, file)
file.close()
file1.close()

print("***** Data in the file input_tensor_data")
file = open('input_tensor_data', 'rb')
data = pickle.load(file)
print("input vectors length",len(data))
file.close()
file1 = open('output_tensor_data', 'rb')
data = pickle.load(file1)
print("output vectors length", len(data))
file1.close()

X_validation, y_validation = [], []
for img in glob.glob('Data/tiffs/*.tiff'):

    image_matrix = np.asarray(Image.open(img))

    image_tensor = torch.from_numpy(image_matrix)

    temp = []
    for i in range(len(image_tensor)):
        for j in range(len(image_tensor[0])):
            temp.append(image_tensor[i][j])

    img = img.split('.')
    img = img[0].split('/')
    img = img[-1].split('_')
    i = position_list.index(int(img[1]))
    if (199<=int(i)<=399 or 599<=int(i)<=799) and int(img[3])<=100:
        X_validation.append(torch.stack(temp))
        y_validation.append(torch.stack([torch.tensor(sheet.cell_value(i + 1, 0)), torch.tensor(sheet.cell_value(i + 1, 1)),
                                     torch.tensor(sheet.cell_value(i + 1, 2))]))
    else:
        input_X.append(torch.stack(temp))
        output_Y.append(torch.stack([torch.tensor(sheet.cell_value(i+1,0)), torch.tensor(sheet.cell_value(i+1,1)),
                                     torch.tensor(sheet.cell_value(i+1,2))]))
file = open('Train_2_input', 'wb')
file1 = open('Train_2_output', 'wb')
pickle.dump(output_Y, file1)
pickle.dump(input_X, file)
file.close()
file1.close()

file = open('Test_2_validation_input', 'wb')
file1 = open('Test_2_validation_output', 'wb')
pickle.dump(y_validation, file1)
pickle.dump(X_validation, file)
file.close()
file1.close()


X_validation, y_validation = [], []
for img in glob.glob('Data/tiffs/*.tiff'):

    image_matrix = np.asarray(Image.open(img))

    image_tensor = torch.from_numpy(image_matrix)

    temp = []
    for i in range(len(image_tensor)):
        for j in range(len(image_tensor[0])):
            temp.append(image_tensor[i][j])

    img = img.split('.')
    img = img[0].split('/')
    img = img[-1].split('_')
    i = position_list.index(int(img[1]))
    if 399<=int(i)<=599:
        X_validation.append(torch.stack(temp))
        y_validation.append(torch.stack([torch.tensor(sheet.cell_value(i + 1, 0)), torch.tensor(sheet.cell_value(i + 1, 1)),
                                     torch.tensor(sheet.cell_value(i + 1, 2))]))
    else:
        input_X.append(torch.stack(temp))
        output_Y.append(torch.stack([torch.tensor(sheet.cell_value(i+1,0)), torch.tensor(sheet.cell_value(i+1,1)),
                                     torch.tensor(sheet.cell_value(i+1,2))]))
file = open('Train_3_input', 'wb')
file1 = open('Train_3_output', 'wb')
pickle.dump(output_Y, file1)
pickle.dump(input_X, file)
file.close()
file1.close()

file = open('Test_3_validation_input', 'wb')
file1 = open('Test_3_validation_output', 'wb')
pickle.dump(y_validation, file1)
pickle.dump(X_validation, file)
file.close()
file1.close()


