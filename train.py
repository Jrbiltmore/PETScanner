import torch
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import dlutils
import numpy as np
from torch.nn import functional as F
from torch import nn

device = torch.cuda.current_device()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
FloatTensor = torch.cuda.FloatTensor
IntTensor = torch.cuda.IntTensor
LongTensor = torch.cuda.LongTensor
print("Running on ", torch.cuda.get_device_name(device))


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(DynamicNet, self).__init__()
        ndf = 128

        self.layer_1 = nn.Conv2d(1, ndf, 4, 2, 1)
        self.layer_2 = nn.Conv2d(ndf, 2 * ndf, 4, 2, 1)
        self.layer_bn_2 = nn.BatchNorm2d(2 * ndf)
        self.layer_3 = nn.Conv2d(2 * ndf, 4 * ndf, 3, 1, 1)
        self.layer_bn_3 = nn.BatchNorm2d(4 * ndf)
        self.layer_4 = nn.Conv2d(4 * ndf, D_out, 4, 1, 0)
        self.y_mean = torch.tensor(np.asarray([3.48624905e+01, - 7.88549283e-01,  1.00120885e-03]), dtype=torch.float32)
        self.y_std = torch.tensor(np.asarray([1.71504586, 4.66643101, 1.02884424]), dtype=torch.float32)

    def forward(self, x):
        x -= 10.86
        x /= 27.45
        x = F.relu(self.layer_1(x[:, None]))
        x = F.relu(self.layer_bn_2(self.layer_2(x)))
        x = F.relu(self.layer_bn_3(self.layer_3(x)))
        x = self.layer_4(x).squeeze()
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

    for i in range(90):
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
