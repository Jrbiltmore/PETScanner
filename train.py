import torch
import pickle
import matplotlib
import matplotlib.pyplot as plt
import math
import dlutils
import numpy as np
from torch.nn import functional as F
from torch import nn
import random
import seaborn as sns


device = torch.cuda.current_device()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
FloatTensor = torch.cuda.FloatTensor
IntTensor = torch.cuda.IntTensor
LongTensor = torch.cuda.LongTensor
print("Running on ", torch.cuda.get_device_name(device))


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class FromGrayScale(nn.Module):
    def __init__(self, channels, outputs):
        super(FromGrayScale, self).__init__()
        self.from_grayscale = nn.Conv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_grayscale(x)
        x = F.relu(x)

        return x


class Block(nn.Module):
    def __init__(self, inputs, outputs, last=False):
        super(Block, self).__init__()
        self.conv_1 = nn.Conv2d(inputs, inputs, 3, 1, 1, bias=False)
        self.blur = Blur(inputs)
        self.last = last
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()
        if last:
            self.dense = nn.Linear(inputs * 4 * 4, outputs)
        else:
            self.conv_2 = nn.Conv2d(inputs, outputs, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inputs)
        self.bn2 = nn.BatchNorm2d(outputs)

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(self.bn1(x), 0.2)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            x = downscale2d(x)
            x = x + self.bias_2
            x = self.bn2(x)
        x = F.relu(x)
        return x


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(DynamicNet, self).__init__()

        inputs = 64
        self.from_grayscale = FromGrayScale(1, inputs)
        self.encode_block: nn.ModuleList[Block] = nn.ModuleList()

        self.encode_block.append(Block(inputs, 2 * inputs, False))

        self.encode_block.append(Block(2 * inputs, 4 * inputs, False))

        self.encode_block.append(Block(4 * inputs, 4 * inputs, True))

        self.fc2 = nn.Linear(4 * inputs, 3)

        self.y_mean = torch.tensor(np.asarray([3.48624905e+01, - 7.88549283e-01,  1.00120885e-03]), dtype=torch.float32)
        self.y_std = torch.tensor(np.asarray([1.71504586, 4.66643101, 1.02884424]), dtype=torch.float32)

    def forward(self, x):
        x -= 10.86
        x /= 27.45
        x = self.from_grayscale(x[:, None])
        x = F.leaky_relu(x, 0.2)

        for i in range(len(self.encode_block)):
            x = self.encode_block[i](x)

        x = self.fc2(x).squeeze()

        return x * self.y_std[None, :] + self.y_mean[None, :]


def load_data(file_x, file_y):
    with open(file_x, 'rb') as f:
        train_X = pickle.load(f)
    with open(file_y, 'rb') as f:
        train_Y = pickle.load(f)

    return list(zip(train_X, train_Y))


def main():
    train = load_data('train_X.pkl', 'train_Y.pkl')
    validation = load_data('validation_X.pkl', 'validation_Y.pkl')

    print(np.asarray([x[0] for x in train]).mean())
    print(np.asarray([x[0] for x in train]).std())

    print(np.asarray([x[1] for x in train]).mean(axis=0))
    print(np.asarray([x[1] for x in train]).std(axis=0))

    #exit()

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_in, H1, H2, D_out = 208, 400, 200, 3

    plotsave = 'Logs/Dropout/Adam/400x200_NoDropout_0-01/'

    model = DynamicNet(D_in, H1, H2, D_out).cuda()

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loss, validation_loss = [], []

    def process(batch):
        x = [x[0] for x in batch]
        y = [x[1] for x in batch]
        x = np.asarray(x)
        x = np.pad(x, ((0, 0), (0, 0), (0, 3)), mode='constant')
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    for i in range(150):
        random.shuffle(train)
        batches = dlutils.batch_provider(train, 128, process)

        for x,  y in batches:
            y_pred = model(x)

            loss = criterion(y_pred, y)

            train_loss.append(math.sqrt(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('loss: %f' % np.asarray(train_loss).mean())

        model.eval()
        with torch.no_grad():
            batches = dlutils.batch_provider(validation, 128, process)

            for x, y in batches:
                y_pred = model(x)

                loss = criterion(y_pred, y)

                validation_loss.append(math.sqrt(loss.item()))

            print('validation_loss: %f' % np.asarray(validation_loss).mean())

        model.train()

    with open(plotsave+'5epochtrain.pkl', 'wb') as f:
        pickle.dump(train_loss, f)
    with open(plotsave+'5epochvalidation.pkl', 'wb') as f:
        pickle.dump(validation_loss, f)
    torch.save(model, plotsave + 'Model20Epochs')

    train_x, train_y = [], []
    # print(validation[0][0], validation[0][1])
    for _ in train:
        train_x.append(_[0])
        train_y.append(_[1])


    validation_x, validation_y = [], []
    #print(validation[0][0], validation[0][1])
    for _ in validation:
        validation_x.append(_[0])
        validation_y.append(_[1])

    print("Lengths of train and validation sets: "+str(len(train_x))+" "+str(len(validation_x)))

    # 100% images in the validation set are in the train set
    validation_1_x = train_x[:len(validation_x)]
    validation_1_y = train_y[:len(validation_y)]
    # 50% images in the validation set are in the train set
    validation_2_x = train_x[:len(validation_x)//2]+validation_x[:len(validation_x)//2]
    validation_2_y = train_y[:len(validation_y)//2]+validation_y[:len(validation_y)//2]
    # 0% images in the validation set are in the train set
    validation_3_x = validation_x
    validation_3_y = validation_y

    validation_1x, validation_1y, validation_1z = [], [], []
    for _ in validation_1_y:
        validation_1x.append(_[0])
        validation_1y.append(_[1])
        validation_1z.append(_[2])

    validation_2x, validation_2y, validation_2z = [], [], []
    for _ in validation_2_y:
        validation_2x.append(_[0])
        validation_2y.append(_[1])
        validation_2z.append(_[2])

    validation_3x, validation_3y, validation_3z = [], [], []
    for _ in validation_3_y:
        validation_3x.append(_[0])
        validation_3y.append(_[1])
        validation_3z.append(_[2])

    trainx, trainy, trainz = [], [], []
    for _ in train_y:
        trainx.append(_[0])
        trainy.append(_[1])
        trainz.append(_[2])
    trainx = np.asarray(trainx)
    trainy = np.asarray(trainy)
    trainz = np.asarray(trainz)
    ax = sns.distplot(trainx)
    ax.set(title='Distribution of X-Coordinate')
    ax.set(xlabel='Normal Distribution M=' + str(trainx.mean()) + ' SD=' + str(trainx.std()),
           ylabel='Frequency')
    plt.show()

    ax = sns.distplot(trainy)
    ax.set(title='Distribution of Y-Coordinate')
    ax.set(xlabel='Normal Distribution M=' + str(trainy.mean()) + ' SD=' + str(trainy.std()), ylabel='Frequency')
    plt.show()

    ax = sns.distplot(trainz)
    ax.set(title='Distribution of Z-Coordinate')
    ax.set(xlabel='Normal Distribution M=' + str(trainz.mean()) + ' SD=' + str(trainz.std()),
           ylabel='Frequency')
    plt.show()



    validation_3x = np.asarray(validation_3x)
    validation_3y = np.asarray(validation_3y)
    validation_3z = np.asarray(validation_3z)
    ax = sns.distplot(validation_3x)
    ax.set(title='Distribution of X-Coordinate')
    ax.set(xlabel='Normal Distribution M=' + str(validation_3x.mean()) + ' SD=' + str(validation_3x.std()), ylabel='Frequency')
    plt.show()

    ax = sns.distplot(validation_3y)
    ax.set(title='Distribution of Y-Coordinate')
    ax.set(xlabel='Normal Distribution M=' + str(validation_3y.mean()) + ' SD=' + str(validation_3y.std()), ylabel='Frequency')
    plt.show()

    ax = sns.distplot(validation_3z)
    ax.set(title='Distribution of Z-Coordinate')
    ax.set(xlabel='Normal Distribution M=' + str(validation_3z.mean()) + ' SD=' + str(validation_3z.std()), ylabel='Frequency')
    plt.show()

    print("*********Validation 3 test case results*********")
    validation_3_x = np.asarray(validation_3_x)
    validation_3_y = np.asarray(validation_3_y)
    validation_3_x = np.asarray(validation_3_x)
    validation_3_x = np.pad(validation_3_x, ((0, 0), (0, 0), (0, 3)), mode='constant')
    validation_3_x = torch.tensor(validation_3_x, dtype=torch.float32)
    validation_3_y = torch.tensor(validation_3_y, dtype=torch.float32)
    model = torch.load(plotsave + 'Model')

    validation_predictions = model(validation_3_x)
    # print("Predicted Validation", validation_predictions)
    # print("Actual Validation", validation_y)
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(validation_predictions, validation_3_y)
    print("RMSE of the validation set: ", math.sqrt(loss))
    errors = abs(validation_predictions - validation_3_y)


    total_errors = []
    for _ in errors:
        total_errors.append(math.sqrt(sum([_[0]**2, _[1]**2, _[2]**2])))
    #print("total errors = ", type(total_errors))
    total_errors = torch.tensor(total_errors)
    total_errors = total_errors.cpu()
    total_errors = total_errors.detach().numpy()


    # print("Mean of the errors:", errors.mean(0))
    # print("Standard Deviation of the errors:", errors.std(0))
    errors = errors.cpu()
    errors = errors.detach().numpy()

    errors_x, errors_y, errors_z = [], [], []
    for _ in errors:
        errors_x.append(_[0])
        errors_y.append(_[1])
        errors_z.append(_[2])


    errors_x = torch.tensor(errors_x, dtype=torch.float32)
    errors_y = torch.tensor(errors_y, dtype=torch.float32)
    errors_z = torch.tensor(errors_z, dtype=torch.float32)
    errors_x = errors_x.cpu()
    errors_y = errors_y.cpu()
    errors_z = errors_z.cpu()

    print("The mean errors on all coordinates: ", errors.mean(0))
    print("The standard deviations on all coordinates: ", errors.std(0))



    # with open(plotsave+'TrainLoss.pkl', 'rb') as f:
    #     train_loss_values = pickle.load(f)
    # with open(plotsave+'ValidationLoss.pkl', 'rb') as f:
    #     validation_loss_values = pickle.load(f)

    # validation_loss_values = torch.tensor(validation_loss_values, dtype=torch.float32)
    # train_loss_values = torch.tensor(train_loss_values, dtype=torch.float32)


    # print("Mean of RMSE of the total validation error: ", validation_loss_values.mean())
    # print("Standard Deviation of the total validation error: ", validation_loss_values.std())
    print("****Total Error Stats****")
    print("Mean of the total error: ", total_errors.mean())
    print("Standard Deviation of the total error: ", total_errors.std())
    print("****X-Coordinate Stats****")
    print("Mean of the error in x-coordinate: ", errors_x.mean())
    print("Standard Deviation of the error in x-coordinate: ", errors_x.std())
    print("****Y-Coordinate Stats****")
    print("Mean of the error in y-coordinate: ", errors_y.mean())
    print("Standard Deviation of the error in y-coordinate: ", errors_y.std())
    print("****Z-Coordinate Stats****")
    print("Mean of the error in z-coordinate: ", errors_z.mean())
    print("Standard Deviation of the error in z-coordinate: ", errors_z.std())

    #print("Total error distribution")
    ax = sns.distplot(total_errors)
    ax.set(title='Distribution of Total Error')
    ax.set(xlabel='Normal Distribution M=' + str(total_errors.mean()) + ' SD=' + str(total_errors.std()), ylabel='Frequency')
    plt.show()

    ax = sns.distplot(errors_x)
    ax.set(title='Distribution of X-Coordinate Errors')
    ax.set(xlabel='Normal Distribution M=' + str(errors_x.mean()) + ' SD=' + str(errors_x.std()), ylabel='Frequency')
    plt.show()

    ax = sns.distplot(errors_y)
    ax.set(title='Distribution of Y-Coordinate Errors')
    ax.set(xlabel='Normal Distribution M=' + str(errors_y.mean()) + ' SD=' + str(errors_y.std()), ylabel='Frequency')
    plt.show()

    ax = sns.distplot(errors_z)
    ax.set(title='Distribution of Z-Coordinate Errors')
    ax.set(xlabel='Normal Distribution M=' + str(errors_z.mean()) + ' SD=' + str(errors_z.std()), ylabel='Frequency')
    plt.show()










    #print("Weight parameters ",list(model.parameters()))
    #
    # file_train_loss = open(plotsave+'Train_Loss', 'wb')
    # pickle.dump(train_loss, file_train_loss)
    # file_train_loss.close()
    # file_validation1_loss = open(plotsave+'Validation1_Loss', 'wb')
    # pickle.dump(validation1_loss, file_validation1_loss)
    # file_validation1_loss.close()
    # file_validation2_loss = open(plotsave+'Validation2_Loss', 'wb')
    # pickle.dump(validation2_loss, file_validation2_loss)
    # file_validation2_loss.close()
    # file_validation3_loss = open(plotsave+'Validation3_Loss', 'wb')
    # pickle.dump(validation3_loss, file_validation3_loss)
    # file_validation3_loss.close()
    #
    # torch.save(model, plotsave+'Model')
    # print("Optimizer =====>>>>" , optimizer)
    #
    #
    #
    # #Stats for all the 3 validation cases
    #
    # model = torch.load(plotsave+"Model")
    #
    # print("Error calculation for the test set TestCase 1")
    # error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
    # percentage_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
    # X_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Y_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Z_percentage_error = torch.tensor([0.], dtype=torch.float)
    # x_coordinate, y_coordinate, z_coordinate = [], [], []
    # x_errors, y_errors, z_errors = [], [], []
    # overall_error = torch.tensor([0.], dtype=torch.float)
    # absolute_error = torch.tensor([0.], dtype=torch.float)
    # # print(model(X_test[100])[0])
    # for _ in range(len(X_test_1)):
    #     predicted_output = model(X_test_1[_])
    #     x_coordinate.append(predicted_output[0])
    #     y_coordinate.append(predicted_output[1])
    #     z_coordinate.append(predicted_output[2])
    #     error_tensor += abs(y_test_1[_] - predicted_output)
    #     X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_test_1[_][0]), abs(y_test_1[_][0])))
    #     Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_test_1[_][1]), abs(y_test_1[_][1])))
    #     Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_test_1[_][2]), abs(y_test_1[_][2])))
    #     x_errors.append(abs(predicted_output[0] - y_test_1[_][0]))
    #     y_errors.append(abs(predicted_output[1] - y_test_1[_][1]))
    #     z_errors.append(abs(predicted_output[2] - y_test_1[_][2]))
    #     overall_error += (abs(predicted_output[0] - y_test_1[_][0]) ** 2 + abs(predicted_output[1] - y_test_1[_][1]) ** 2 + abs(
    #             predicted_output[2] - y_test_1[_][2]) ** 2)
    #     absolute_error += (abs(predicted_output[0] - y_test_1[_][0]) + abs(predicted_output[1] - y_test_1[_][1]) + abs(
    #             predicted_output[2] - y_test_1[_][2]))
    #     # print(_)
    # X_outputs = torch.stack(x_coordinate)
    # Y_outputs = torch.stack(y_coordinate)
    # Z_outputs = torch.stack(z_coordinate)
    # X_errors = torch.stack(x_errors)
    # Y_errors = torch.stack(y_errors)
    # Z_errors = torch.stack(z_errors)
    # X_errors = X_errors.cpu()
    # Y_errors = Y_errors.cpu()
    # Z_errors = Z_errors.cpu()
    # X_errors = X_errors.detach().numpy()
    # Y_errors = Y_errors.detach().numpy()
    # Z_errors = Z_errors.detach().numpy()
    # error = error_tensor.div(len(X_test_1))
    # print("Total error TestCase 1 ==> ", error_tensor)
    # #print("Mean error of the test set TestCase 1 ==> ", error_tensor.div(len(X_test_1)))
    # print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
    # print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
    # print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
    # print("X Percentage error of the test set TestCase 1 ==> ", X_percentage_error.div(len(X_test_1)))
    # print("Y Percentage error of the test set TestCase 1 ==> ", Y_percentage_error.div(len(X_test_1)))
    # print("Z Percentage error of the test set TestCase 1 ==> ", Z_percentage_error.div(len(X_test_1)))
    # print("Mean error of the test set TestCase 1 X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
    # print("Mean error of the test set TestCase 1 Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
    # print("Mean error of the test set TestCase 1 Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
    # print("Overall error of a point of the test set TestCase 1 ==> ", torch.sqrt(overall_error.div(len(X_test_1))))
    # print("Absolute error of a point of the test set TestCase 1 ==> ", absolute_error.div(len(X_test_1)))
    #
    #
    # X_outputs = X_outputs.cpu()
    # Y_outputs = Y_outputs.cpu()
    # Z_outputs = Z_outputs.cpu()
    # X_outputs = X_outputs.detach().numpy()
    # Y_outputs = Y_outputs.detach().numpy()
    # Z_outputs = Z_outputs.detach().numpy()
    #
    # error = error.cpu()
    # error = error.detach().numpy()
    #
    # ax = sns.distplot(X_errors)
    # ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Y_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Z_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
    # plt.show()
    # #plt.hist(X_outputs, edgecolor='black')
    # #plt.show()
    # #plt.savefig(plotsave + 'test_1_x.png')
    #
    # #plt.hist(Y_outputs, edgecolor='black')
    # #plt.show()
    # #plt2.savefig(plotsave + 'test_1_y.png')
    #
    # #plt.hist(Z_outputs, edgecolor='black')
    # #plt.show()
    # #plt3.savefig(plotsave + 'test_1_z.png')
    #
    # print("***************************************************")
    #
    # print("Error calculation for test set TestCase 2 i.e., 50% images from the train set")
    # error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
    #
    # x_coordinate, y_coordinate, z_coordinate = [], [], []
    # x_errors, y_errors, z_errors = [], [], []
    # X_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Y_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Z_percentage_error = torch.tensor([0.], dtype=torch.float)
    # overall_error = torch.tensor([0.], dtype=torch.float)
    # absolute_error = torch.tensor([0.], dtype=torch.float)
    # for _ in range(len(X_test_2)):
    #     predicted_output = model(X_test_2[_])
    #     x_coordinate.append(predicted_output[0])
    #     y_coordinate.append(predicted_output[1])
    #     z_coordinate.append(predicted_output[2])
    #     error_tensor += abs(y_test_2[_] - predicted_output)
    #     X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_test_2[_][0]), abs(y_test_2[_][0])))
    #     Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_test_2[_][1]), abs(y_test_2[_][1])))
    #     Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_test_2[_][2]), abs(y_test_2[_][2])))
    #     x_errors.append(abs(predicted_output[0] - y_test_1[_][0]))
    #     y_errors.append(abs(predicted_output[1] - y_test_1[_][1]))
    #     z_errors.append(abs(predicted_output[2] - y_test_1[_][2]))
    #     overall_error += abs(predicted_output[0] - y_test_2[_][0]) ** 2 + abs(predicted_output[1] - y_test_2[_][1]) ** 2 + abs(
    #             predicted_output[2] - y_test_2[_][2]) ** 2
    #     absolute_error += abs(predicted_output[0] - y_test_2[_][0]) + abs(
    #         predicted_output[1] - y_test_2[_][1]) + abs(
    #         predicted_output[2] - y_test_2[_][2])
    #     # print(_)
    # X_outputs = torch.stack(x_coordinate)
    # Y_outputs = torch.stack(y_coordinate)
    # Z_outputs = torch.stack(z_coordinate)
    # error = error_tensor.div(len(X_test_2))
    # X_errors = torch.stack(x_errors)
    # Y_errors = torch.stack(y_errors)
    # Z_errors = torch.stack(z_errors)
    # X_errors = X_errors.cpu()
    # Y_errors = Y_errors.cpu()
    # Z_errors = Z_errors.cpu()
    # X_errors = X_errors.detach().numpy()
    # Y_errors = Y_errors.detach().numpy()
    # Z_errors = Z_errors.detach().numpy()
    # print("Total error TestCase 2 ==> ", error_tensor)
    # #print("Mean error of the test set TestCase 2 ==> ", error_tensor.div(len(X_test_2)))
    # print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
    # print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
    # print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
    # print("X Percentage error of the test set TestCase 2 ==> ", X_percentage_error.div(len(X_test_2)))
    # print("Y Percentage error of the test set TestCase 2 ==> ", Y_percentage_error.div(len(X_test_2)))
    # print("Z Percentage error of the test set TestCase 2 ==> ", Z_percentage_error.div(len(X_test_2)))
    # print("Mean error of the test set TestCase 2 X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
    # print("Mean error of the test set TestCase 2 Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
    # print("Mean error of the test set TestCase 2 Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
    # print("Overall error of a point of the test set TestCase 2 ==> ", torch.sqrt(overall_error.div(len(X_test_2))))
    # print("Absolute Mean error of a point of the test set TestCase 2 ==> ", absolute_error.div(len(X_test_2)))
    #
    # X_outputs = X_outputs.cpu()
    # Y_outputs = Y_outputs.cpu()
    # Z_outputs = Z_outputs.cpu()
    #
    # X_outputs = X_outputs.detach().numpy()
    # Y_outputs = Y_outputs.detach().numpy()
    # Z_outputs = Z_outputs.detach().numpy()
    #
    #
    # ax = sns.distplot(X_errors)
    # ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Y_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Z_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # #plt.hist(X_outputs, edgecolor='black')
    # #plt.show()
    # #plt.savefig(plotsave + 'test_1_x.png')
    #
    # #plt.hist(Y_outputs, edgecolor='black')
    # #plt.show()
    # #plt2.savefig(plotsave + 'test_1_y.png')
    #
    # #plt.hist(Z_outputs, edgecolor='black')
    # #plt.show()
    # #plt3.savefig(plotsave + 'test_1_z.png')
    #
    #
    # print("***************************************************")
    #
    # print("Error calculation for test set TestCase 3 i.e., 100% images from the train set")
    # error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
    # percentage_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
    # # X_test_3 = X[:43377]
    # # y_test_3 = y[:43377]
    # X_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Y_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Z_percentage_error = torch.tensor([0.], dtype=torch.float)
    # x_coordinate, y_coordinate, z_coordinate = [], [], []
    # x_errors, y_errors, z_errors = [], [], []
    # overall_error = torch.tensor([0.], dtype=torch.float)
    # absolute_error = torch.tensor([0.], dtype=torch.float)
    # for _ in range(len(X_test_3)):
    #     predicted_output = model(X_test_3[_])
    #     x_coordinate.append(predicted_output[0])
    #     y_coordinate.append(predicted_output[1])
    #     z_coordinate.append(predicted_output[2])
    #     error_tensor += abs(y_test_3[_] - predicted_output)
    #     X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_test_3[_][0]), abs(y_test_3[_][0])))
    #     Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_test_3[_][1]), abs(y_test_3[_][1])))
    #     Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_test_3[_][2]), abs(y_test_3[_][2])))
    #     x_errors.append(abs(predicted_output[0] - y_test_1[_][0]))
    #     y_errors.append(abs(predicted_output[1] - y_test_1[_][1]))
    #     z_errors.append(abs(predicted_output[2] - y_test_1[_][2]))
    #     overall_error += abs(predicted_output[0] - y_test_3[_][0]) ** 2 + abs(predicted_output[1] - y_test_3[_][1]) ** 2 + abs(
    #             predicted_output[2] - y_test_3[_][2]) ** 2
    #     absolute_error += abs(predicted_output[0] - y_test_3[_][0]) + abs(
    #         predicted_output[1] - y_test_3[_][1]) + abs(
    #         predicted_output[2] - y_test_3[_][2])
    #     # print(_)
    # X_outputs = torch.stack(x_coordinate)
    # Y_outputs = torch.stack(y_coordinate)
    # Z_outputs = torch.stack(z_coordinate)
    # error = error_tensor.div(len(X_test_3))
    # X_errors = torch.stack(x_errors)
    # Y_errors = torch.stack(y_errors)
    # Z_errors = torch.stack(z_errors)
    # X_errors = X_errors.cpu()
    # Y_errors = Y_errors.cpu()
    # Z_errors = Z_errors.cpu()
    # X_errors = X_errors.detach().numpy()
    # Y_errors = Y_errors.detach().numpy()
    # Z_errors = Z_errors.detach().numpy()
    # print("Total error TestCase 3 ==> ", error_tensor)
    # #print("Mean error of the test set TestCase 3 ==> ", error_tensor.div(len(X_test_3)))
    # print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
    # print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
    # print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
    # print("X Percentage error of the test set TestCase 3 ==> ", X_percentage_error.div(len(X_test_3)))
    # print("Y Percentage error of the test set TestCase 3 ==> ", Y_percentage_error.div(len(X_test_3)))
    # print("Z Percentage error of the test set TestCase 3 ==> ", Z_percentage_error.div(len(X_test_3)))
    # print("Mean error of the test set TestCase 3 X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
    # print("Mean error of the test set TestCase 3 Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
    # print("Mean error of the test set TestCase 3 Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
    # print("Overall error of a point of the test set TestCase 3 ==> ", torch.sqrt(overall_error.div(len(X_test_3))))
    # print("Absolute Mean error of a point of the test set TestCase 3 ==> ", absolute_error.div(len(X_test_3)))
    #
    # X_outputs = X_outputs.cpu()
    # Y_outputs = Y_outputs.cpu()
    # Z_outputs = Z_outputs.cpu()
    # X_outputs = X_outputs.detach().numpy()
    # Y_outputs = Y_outputs.detach().numpy()
    # Z_outputs = Z_outputs.detach().numpy()
    #
    #
    # ax = sns.distplot(X_errors)
    # ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Y_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Z_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
    # plt.show()
    # #plt.hist(X_outputs, edgecolor='black')
    # #plt.show()
    # #plt.savefig(plotsave + 'test_1_x.png')
    #
    # #plt.hist(Y_outputs, edgecolor='black')
    # #plt.show()
    # #plt2.savefig(plotsave + 'test_1_y.png')
    #
    # #plt.hist(Z_outputs, edgecolor='black')
    # #plt.show()
    # #plt3.savefig(plotsave + 'test_1_z.png')
    #
    #
    # print("***************************************************")
    #
    # print("Error calculation for Train Set ")
    # error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
    # percentage_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
    # # X_test_3 = X[:43377]
    # # y_test_3 = y[:43377]
    # X_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Y_percentage_error = torch.tensor([0.], dtype=torch.float)
    # Z_percentage_error = torch.tensor([0.], dtype=torch.float)
    # x_coordinate, y_coordinate, z_coordinate = [], [], []
    # x_errors, y_errors, z_errors = [], [], []
    # overall_error = torch.tensor([0.], dtype=torch.float)
    # absolute_error = torch.tensor([0.], dtype=torch.float)
    # for _ in range(len(X_train)):
    #     predicted_output = model(X_train[_])
    #     x_coordinate.append(predicted_output[0])
    #     y_coordinate.append(predicted_output[1])
    #     z_coordinate.append(predicted_output[2])
    #     error_tensor += abs(y_train[_] - predicted_output)
    #     X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_train[_][0]), abs(y_train[_][0])))
    #     Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_train[_][1]), abs(y_train[_][1])))
    #     Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_train[_][2]), abs(y_train[_][2])))
    #     x_errors.append(abs(predicted_output[0] - y_train[_][0]))
    #     y_errors.append(abs(predicted_output[1] - y_train[_][1]))
    #     z_errors.append(abs(predicted_output[2] - y_train[_][2]))
    #     overall_error += abs(predicted_output[0] - y_train[_][0]) ** 2 + abs(predicted_output[1] - y_train[_][1]) ** 2 + abs(
    #             predicted_output[2] - y_train[_][2]) ** 2
    #     overall_error += abs(predicted_output[0] - y_train[_][0]) + abs(
    #         predicted_output[1] - y_train[_][1]) + abs(
    #         predicted_output[2] - y_train[_][2])
    #     # print(_)
    # X_outputs = torch.stack(x_coordinate)
    # Y_outputs = torch.stack(y_coordinate)
    # Z_outputs = torch.stack(z_coordinate)
    # error = error_tensor.div(len(X_train))
    # X_errors = torch.stack(x_errors)
    # Y_errors = torch.stack(y_errors)
    # Z_errors = torch.stack(z_errors)
    # X_errors = X_errors.cpu()
    # Y_errors = Y_errors.cpu()
    # Z_errors = Z_errors.cpu()
    # X_errors = X_errors.detach().numpy()
    # Y_errors = Y_errors.detach().numpy()
    # Z_errors = Z_errors.detach().numpy()
    # print("Total error Train set ==> ", error_tensor)
    # #print("Mean error of the Train set ==> ", error_tensor.div(len(X_train)))
    # print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
    # print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
    # print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
    # print("X Percentage error of the Train set ==> ", X_percentage_error.div(len(X_train)))
    # print("Y Percentage error of the Train set ==> ", Y_percentage_error.div(len(X_train)))
    # print("Z Percentage error of the Train set ==> ", Z_percentage_error.div(len(X_train)))
    # print("Mean error of the Train Set X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
    # print("Mean error of the Train Set Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
    # print("Mean error of the Train Set Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
    # print("Overall error of a point of the Train set ==> ", torch.sqrt(overall_error.div(len(X_train))))
    # print("Absolute Mean error of a point of the Train set ==> ", absolute_error.div(len(X_train)))
    #
    # X_outputs = X_outputs.cpu()
    # Y_outputs = Y_outputs.cpu()
    # Z_outputs = Z_outputs.cpu()
    # X_outputs = X_outputs.detach().numpy()
    # Y_outputs = Y_outputs.detach().numpy()
    # Z_outputs = Z_outputs.detach().numpy()
    #
    #
    # ax = sns.distplot(X_errors)
    # ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Y_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
    # plt.show()
    #
    # ax = sns.distplot(Z_errors)
    # ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
    # plt.show()
    # #plt.hist(X_outputs, edgecolor='black')
    # #plt.show()
    # #plt.savefig(plotsave + 'test_1_x.png')
    #
    # #plt.hist(Y_outputs, edgecolor='black')
    # #plt.show()
    # #plt2.savefig(plotsave + 'test_1_y.png')
    #
    # #plt.hist(Z_outputs, edgecolor='black')
    # #plt.show()
    # #plt3.savefig(plotsave + 'test_1_z.png')
    #
    #
    #
    #
    #
    #
    #
    # # number of parameters criteria
    # print("*********************************************")
    # print("The 2n+d number ==> ", 2 * len(X_train) + len(X_train[0]))
    # print("The actual number of parameters ==> ", D_in * H1 + H1 * H2 + H2 * D_out)
    #
    # file_train_loss = open(plotsave + 'Train_Loss', 'rb')
    # train_loss_values = pickle.load(file_train_loss)
    # file_train_loss.close()
    #
    # file_validation1_loss = open(plotsave + 'Validation1_Loss', 'rb')
    # validation1_loss_values = pickle.load(file_validation1_loss)
    # file_validation1_loss.close()
    #
    # file_validation2_loss = open(plotsave + 'Validation2_Loss', 'rb')
    # validation2_loss_values = pickle.load(file_validation2_loss)
    # file_validation2_loss.close()
    #
    # file_validation3_loss = open(plotsave + 'Validation3_Loss', 'rb')
    # validation3_loss_values = pickle.load(file_validation3_loss)
    # file_validation3_loss.close()
    #
    # plt.plot(train_loss_values, label='Train Loss')
    # plt.plot(validation1_loss_values, label='Validation1 Loss')
    # plt.plot(validation2_loss_values, label='Validation2 Loss')
    # plt.plot(validation3_loss_values, label='Validation3 Loss')
    # plt.ylabel('Loss Values')
    # plt.xlabel('Iterations')
    #
    # #plt.legend('Train Loss', 'Validation1 Loss', 'Validation2 Loss', 'Validation3 Loss')
    # labels = ['Train Loss', 'Validation1 Loss', 'Validation2 Loss', 'Validation3 Loss']
    # plt.legend(labels)
    # plt.xlim(0, 100)
    # plt.ylim(0, 200)
    #
    # plt.show()
    # #plt10.savefig(plotsave + 'loss.png')
    #
    # trainLoss_errors, validation1Loss_errors, validation2Loss_errors, validation3Loss_errors = [], [], [], []
    # file_trainLossError = open(plotsave+'Train_Loss', 'rb')
    # train_error = pickle.load(file_trainLossError)
    # file_trainLossError.close()
    #
    # print("Train error values", train_error)
    # ax = sns.distplot(train_error)
    # ax.set(xlabel='Normal Distribution', ylabel='Frequency')
    # plt.show()
    # #file_validation1LossError = open(plotsave+'Validation1_Loss', rb)


if __name__ == "__main__":
    main()

