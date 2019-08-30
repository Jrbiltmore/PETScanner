# Code in file nn/two_layer_net_nn.py
'''
import torch

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to(device)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function. Setting
# reduction='sum' means that we are computing the *sum* of squared errors rather
# than the mean; this is for consistency with the examples above where we
# manually compute the loss, but in practice it is more common to use mean
# squared error as a loss by setting reduction='elementwise_mean'.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its data and gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad
'''




# Data import
import torch
import pickle

file = open('Train_1_input', 'rb')
file1 = open('Train_1_output', 'rb')
input = pickle.load(file)
output = pickle.load(file1)

file.close()
file1.close()

# N, D_in, H, D_out = 60000, 208, 600, 3

X_train, y_train = [], []

for i in range(len(input)):
    X_train.append(input[i])
    y_train.append(output[i])

X_train = torch.stack(X_train)
y_train = torch.stack(y_train)
X_train = X_train.float()
y_train = y_train.float()


use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

# If zd_merge true, will use zd discriminator that looks at entire batch.
zd_merge = False

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))

    X_train = X_train.cuda()
    y_train = y_train.cuda()




# Code in file nn/dynamic_net.py
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H1)
        self.hidden1_linear = torch.nn.Linear(H1, H2)
        self.hidden2_linear = torch.nn.Linear(H2, H2)
        self.output_linear = torch.nn.Linear(H2, D_out)
        self.dropout = torch.nn.Dropout(p=0.6)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        # self.dropout = torch.nn.Dropout(p=0.2)
        h_relu = self.input_linear(x).clamp(min=0)
        #temp = self.dropout(h_relu)
        # print("output from the input layer ", h_relu)
        h_relu = self.hidden1_linear(h_relu).clamp(min=0)
        # print("H_RELU values ", h_relu)
        #temp = self.dropout(h_re+
        # u)
        # print("TEMP values ", temp)
        h_relu = self.hidden2_linear(h_relu).clamp(min=0)
        #temp = self.dropout(h_relu)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H1, H2, D_out = 208, 400, 200, 3


# Data preparation
file = open('Test_1_validation_input', 'rb')
file1 = open('Test_1_validation_output', 'rb')
input = pickle.load(file)
output = pickle.load(file1)

file.close()
file1.close()

# N, D_in, H, D_out = 60000, 208, 600, 3

X_test, y_test = [], []

for i in range(len(input)):
    X_test.append(input[i])
    y_test.append(output[i])

X_test_1 = torch.stack(X_test)
y_test_1 = torch.stack(y_test)
X_test_1 = X_test_1.float()
y_test_1 = y_test_1.float()

file = open('Test_3_validation_input', 'rb')
file1 = open('Test_3_validation_output', 'rb')
input = pickle.load(file)
output = pickle.load(file1)

file.close()
file1.close()

# N, D_in, H, D_out = 60000, 208, 600, 3

X_test, y_test = [], []

for i in range(len(input)):
    X_test.append(input[i])
    y_test.append(output[i])

X_test_3 = torch.stack(X_test)
y_test_3 = torch.stack(y_test)
X_test_3 = X_test_3.float()
y_test_3 = y_test_3.float()

file = open('Test_2_validation_input', 'rb')
file1 = open('Test_2_validation_output', 'rb')
input = pickle.load(file)
output = pickle.load(file1)

file.close()
file1.close()

# N, D_in, H, D_out = 60000, 208, 600, 3

X_test, y_test = [], []

for i in range(len(input)):
    X_test.append(input[i])
    y_test.append(output[i])

X_test_2 = torch.stack(X_test)
y_test_2 = torch.stack(y_test)
X_test_2 = X_test_2.float()
y_test_2 = y_test_2.float()


if use_cuda:
    X_test_1 = X_test_1.cuda()
    X_test_2 = X_test_2.cuda()
    X_test_3 = X_test_3.cuda()
    y_test_1 = y_test_1.cuda()
    y_test_2 = y_test_2.cuda()
    y_test_3 = y_test_3.cuda()


plotsave = 'Logs/Dropout/Adam/400x200_NoDropout_0-01/'



# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H1, H2, D_out)

if use_cuda:
    model.cuda()

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='mean') #if Mean squared, reduction = 'elementwise_mean'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
train_loss, validation1_loss, validation2_loss, validation3_loss = [], [], [], []

import math
for t in range(9000):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(X_train)

  # Compute and print loss
  loss = criterion(y_pred, y_train)
  print(t, math.sqrt(loss.item()))
  train_loss.append(math.sqrt(loss.item()))

  #print(temp_y_test_1.item(), temp_y_test_2.item(), temp_y_test_3.item())

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if t % 100 == 0:
      model.eval()
      with torch.no_grad():
          temp_y_test_2 = model(X_test_2)
          loss_2 = criterion(temp_y_test_2, y_test_2)
          validation2_loss.append(math.sqrt(loss_2.item()))

          temp_y_test_1 = model(X_test_1)
          loss_1 = criterion(temp_y_test_1, y_test_1)
          validation1_loss.append(math.sqrt(loss_1.item()))

          temp_y_test_3 = model(X_test_3)
          loss_3 = criterion(temp_y_test_3, y_test_3)
          validation3_loss.append(math.sqrt(loss_3.item()))

          print(t, math.sqrt(loss_1.item()), math.sqrt(loss_2.item()), math.sqrt(loss_3.item()))

      model.train()

#print("Weight parameters ",list(model.parameters()))

file_train_loss = open(plotsave+'Train_Loss', 'wb')
pickle.dump(train_loss, file_train_loss)
file_train_loss.close()
file_validation1_loss = open(plotsave+'Validation1_Loss', 'wb')
pickle.dump(validation1_loss, file_validation1_loss)
file_validation1_loss.close()
file_validation2_loss = open(plotsave+'Validation2_Loss', 'wb')
pickle.dump(validation2_loss, file_validation2_loss)
file_validation2_loss.close()
file_validation3_loss = open(plotsave+'Validation3_Loss', 'wb')
pickle.dump(validation3_loss, file_validation3_loss)
file_validation3_loss.close()

torch.save(model, plotsave+'Model')
print("Optimizer =====>>>>" , optimizer)



#Stats for all the 3 validation cases

model = torch.load(plotsave+"Model")

print("Error calculation for the test set TestCase 1")
error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
percentage_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
X_percentage_error = torch.tensor([0.], dtype=torch.float)
Y_percentage_error = torch.tensor([0.], dtype=torch.float)
Z_percentage_error = torch.tensor([0.], dtype=torch.float)
x_coordinate, y_coordinate, z_coordinate = [], [], []
x_errors, y_errors, z_errors = [], [], []
overall_error = torch.tensor([0.], dtype=torch.float)
absolute_error = torch.tensor([0.], dtype=torch.float)
# print(model(X_test[100])[0])
for _ in range(len(X_test_1)):
    predicted_output = model(X_test_1[_])
    x_coordinate.append(predicted_output[0])
    y_coordinate.append(predicted_output[1])
    z_coordinate.append(predicted_output[2])
    error_tensor += abs(y_test_1[_] - predicted_output)
    X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_test_1[_][0]), abs(y_test_1[_][0])))
    Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_test_1[_][1]), abs(y_test_1[_][1])))
    Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_test_1[_][2]), abs(y_test_1[_][2])))
    x_errors.append(abs(predicted_output[0] - y_test_1[_][0]))
    y_errors.append(abs(predicted_output[1] - y_test_1[_][1]))
    z_errors.append(abs(predicted_output[2] - y_test_1[_][2]))
    overall_error += (abs(predicted_output[0] - y_test_1[_][0]) ** 2 + abs(predicted_output[1] - y_test_1[_][1]) ** 2 + abs(
            predicted_output[2] - y_test_1[_][2]) ** 2)
    absolute_error += (abs(predicted_output[0] - y_test_1[_][0]) + abs(predicted_output[1] - y_test_1[_][1]) + abs(
            predicted_output[2] - y_test_1[_][2]))
    # print(_)
X_outputs = torch.stack(x_coordinate)
Y_outputs = torch.stack(y_coordinate)
Z_outputs = torch.stack(z_coordinate)
X_errors = torch.stack(x_errors)
Y_errors = torch.stack(y_errors)
Z_errors = torch.stack(z_errors)
X_errors = X_errors.cpu()
Y_errors = Y_errors.cpu()
Z_errors = Z_errors.cpu()
X_errors = X_errors.detach().numpy()
Y_errors = Y_errors.detach().numpy()
Z_errors = Z_errors.detach().numpy()
error = error_tensor.div(len(X_test_1))
print("Total error TestCase 1 ==> ", error_tensor)
#print("Mean error of the test set TestCase 1 ==> ", error_tensor.div(len(X_test_1)))
print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
print("X Percentage error of the test set TestCase 1 ==> ", X_percentage_error.div(len(X_test_1)))
print("Y Percentage error of the test set TestCase 1 ==> ", Y_percentage_error.div(len(X_test_1)))
print("Z Percentage error of the test set TestCase 1 ==> ", Z_percentage_error.div(len(X_test_1)))
print("Mean error of the test set TestCase 1 X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
print("Mean error of the test set TestCase 1 Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
print("Mean error of the test set TestCase 1 Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
print("Overall error of a point of the test set TestCase 1 ==> ", torch.sqrt(overall_error.div(len(X_test_1))))
print("Absolute error of a point of the test set TestCase 1 ==> ", absolute_error.div(len(X_test_1)))

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

X_outputs = X_outputs.cpu()
Y_outputs = Y_outputs.cpu()
Z_outputs = Z_outputs.cpu()
X_outputs = X_outputs.detach().numpy()
Y_outputs = Y_outputs.detach().numpy()
Z_outputs = Z_outputs.detach().numpy()

error = error.cpu()
error = error.detach().numpy()

ax = sns.distplot(X_errors)
ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Y_errors)
ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Z_errors)
ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
plt.show()
#plt.hist(X_outputs, edgecolor='black')
#plt.show()
#plt.savefig(plotsave + 'test_1_x.png')

#plt.hist(Y_outputs, edgecolor='black')
#plt.show()
#plt2.savefig(plotsave + 'test_1_y.png')

#plt.hist(Z_outputs, edgecolor='black')
#plt.show()
#plt3.savefig(plotsave + 'test_1_z.png')

print("***************************************************")

print("Error calculation for test set TestCase 2 i.e., 50% images from the train set")
error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)

x_coordinate, y_coordinate, z_coordinate = [], [], []
x_errors, y_errors, z_errors = [], [], []
X_percentage_error = torch.tensor([0.], dtype=torch.float)
Y_percentage_error = torch.tensor([0.], dtype=torch.float)
Z_percentage_error = torch.tensor([0.], dtype=torch.float)
overall_error = torch.tensor([0.], dtype=torch.float)
absolute_error = torch.tensor([0.], dtype=torch.float)
for _ in range(len(X_test_2)):
    predicted_output = model(X_test_2[_])
    x_coordinate.append(predicted_output[0])
    y_coordinate.append(predicted_output[1])
    z_coordinate.append(predicted_output[2])
    error_tensor += abs(y_test_2[_] - predicted_output)
    X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_test_2[_][0]), abs(y_test_2[_][0])))
    Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_test_2[_][1]), abs(y_test_2[_][1])))
    Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_test_2[_][2]), abs(y_test_2[_][2])))
    x_errors.append(abs(predicted_output[0] - y_test_1[_][0]))
    y_errors.append(abs(predicted_output[1] - y_test_1[_][1]))
    z_errors.append(abs(predicted_output[2] - y_test_1[_][2]))
    overall_error += abs(predicted_output[0] - y_test_2[_][0]) ** 2 + abs(predicted_output[1] - y_test_2[_][1]) ** 2 + abs(
            predicted_output[2] - y_test_2[_][2]) ** 2
    absolute_error += abs(predicted_output[0] - y_test_2[_][0]) + abs(
        predicted_output[1] - y_test_2[_][1]) + abs(
        predicted_output[2] - y_test_2[_][2])
    # print(_)
X_outputs = torch.stack(x_coordinate)
Y_outputs = torch.stack(y_coordinate)
Z_outputs = torch.stack(z_coordinate)
error = error_tensor.div(len(X_test_2))
X_errors = torch.stack(x_errors)
Y_errors = torch.stack(y_errors)
Z_errors = torch.stack(z_errors)
X_errors = X_errors.cpu()
Y_errors = Y_errors.cpu()
Z_errors = Z_errors.cpu()
X_errors = X_errors.detach().numpy()
Y_errors = Y_errors.detach().numpy()
Z_errors = Z_errors.detach().numpy()
print("Total error TestCase 2 ==> ", error_tensor)
#print("Mean error of the test set TestCase 2 ==> ", error_tensor.div(len(X_test_2)))
print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
print("X Percentage error of the test set TestCase 2 ==> ", X_percentage_error.div(len(X_test_2)))
print("Y Percentage error of the test set TestCase 2 ==> ", Y_percentage_error.div(len(X_test_2)))
print("Z Percentage error of the test set TestCase 2 ==> ", Z_percentage_error.div(len(X_test_2)))
print("Mean error of the test set TestCase 2 X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
print("Mean error of the test set TestCase 2 Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
print("Mean error of the test set TestCase 2 Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
print("Overall error of a point of the test set TestCase 2 ==> ", torch.sqrt(overall_error.div(len(X_test_2))))
print("Absolute Mean error of a point of the test set TestCase 2 ==> ", absolute_error.div(len(X_test_2)))

X_outputs = X_outputs.cpu()
Y_outputs = Y_outputs.cpu()
Z_outputs = Z_outputs.cpu()

X_outputs = X_outputs.detach().numpy()
Y_outputs = Y_outputs.detach().numpy()
Z_outputs = Z_outputs.detach().numpy()


ax = sns.distplot(X_errors)
ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Y_errors)
ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Z_errors)
ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
plt.show()

#plt.hist(X_outputs, edgecolor='black')
#plt.show()
#plt.savefig(plotsave + 'test_1_x.png')

#plt.hist(Y_outputs, edgecolor='black')
#plt.show()
#plt2.savefig(plotsave + 'test_1_y.png')

#plt.hist(Z_outputs, edgecolor='black')
#plt.show()
#plt3.savefig(plotsave + 'test_1_z.png')


print("***************************************************")

print("Error calculation for test set TestCase 3 i.e., 100% images from the train set")
error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
percentage_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
# X_test_3 = X[:43377]
# y_test_3 = y[:43377]
X_percentage_error = torch.tensor([0.], dtype=torch.float)
Y_percentage_error = torch.tensor([0.], dtype=torch.float)
Z_percentage_error = torch.tensor([0.], dtype=torch.float)
x_coordinate, y_coordinate, z_coordinate = [], [], []
x_errors, y_errors, z_errors = [], [], []
overall_error = torch.tensor([0.], dtype=torch.float)
absolute_error = torch.tensor([0.], dtype=torch.float)
for _ in range(len(X_test_3)):
    predicted_output = model(X_test_3[_])
    x_coordinate.append(predicted_output[0])
    y_coordinate.append(predicted_output[1])
    z_coordinate.append(predicted_output[2])
    error_tensor += abs(y_test_3[_] - predicted_output)
    X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_test_3[_][0]), abs(y_test_3[_][0])))
    Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_test_3[_][1]), abs(y_test_3[_][1])))
    Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_test_3[_][2]), abs(y_test_3[_][2])))
    x_errors.append(abs(predicted_output[0] - y_test_1[_][0]))
    y_errors.append(abs(predicted_output[1] - y_test_1[_][1]))
    z_errors.append(abs(predicted_output[2] - y_test_1[_][2]))
    overall_error += abs(predicted_output[0] - y_test_3[_][0]) ** 2 + abs(predicted_output[1] - y_test_3[_][1]) ** 2 + abs(
            predicted_output[2] - y_test_3[_][2]) ** 2
    absolute_error += abs(predicted_output[0] - y_test_3[_][0]) + abs(
        predicted_output[1] - y_test_3[_][1]) + abs(
        predicted_output[2] - y_test_3[_][2])
    # print(_)
X_outputs = torch.stack(x_coordinate)
Y_outputs = torch.stack(y_coordinate)
Z_outputs = torch.stack(z_coordinate)
error = error_tensor.div(len(X_test_3))
X_errors = torch.stack(x_errors)
Y_errors = torch.stack(y_errors)
Z_errors = torch.stack(z_errors)
X_errors = X_errors.cpu()
Y_errors = Y_errors.cpu()
Z_errors = Z_errors.cpu()
X_errors = X_errors.detach().numpy()
Y_errors = Y_errors.detach().numpy()
Z_errors = Z_errors.detach().numpy()
print("Total error TestCase 3 ==> ", error_tensor)
#print("Mean error of the test set TestCase 3 ==> ", error_tensor.div(len(X_test_3)))
print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
print("X Percentage error of the test set TestCase 3 ==> ", X_percentage_error.div(len(X_test_3)))
print("Y Percentage error of the test set TestCase 3 ==> ", Y_percentage_error.div(len(X_test_3)))
print("Z Percentage error of the test set TestCase 3 ==> ", Z_percentage_error.div(len(X_test_3)))
print("Mean error of the test set TestCase 3 X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
print("Mean error of the test set TestCase 3 Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
print("Mean error of the test set TestCase 3 Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
print("Overall error of a point of the test set TestCase 3 ==> ", torch.sqrt(overall_error.div(len(X_test_3))))
print("Absolute Mean error of a point of the test set TestCase 3 ==> ", absolute_error.div(len(X_test_3)))

X_outputs = X_outputs.cpu()
Y_outputs = Y_outputs.cpu()
Z_outputs = Z_outputs.cpu()
X_outputs = X_outputs.detach().numpy()
Y_outputs = Y_outputs.detach().numpy()
Z_outputs = Z_outputs.detach().numpy()


ax = sns.distplot(X_errors)
ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Y_errors)
ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Z_errors)
ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
plt.show()
#plt.hist(X_outputs, edgecolor='black')
#plt.show()
#plt.savefig(plotsave + 'test_1_x.png')

#plt.hist(Y_outputs, edgecolor='black')
#plt.show()
#plt2.savefig(plotsave + 'test_1_y.png')

#plt.hist(Z_outputs, edgecolor='black')
#plt.show()
#plt3.savefig(plotsave + 'test_1_z.png')


print("***************************************************")

print("Error calculation for Train Set ")
error_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
percentage_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)
# X_test_3 = X[:43377]
# y_test_3 = y[:43377]
X_percentage_error = torch.tensor([0.], dtype=torch.float)
Y_percentage_error = torch.tensor([0.], dtype=torch.float)
Z_percentage_error = torch.tensor([0.], dtype=torch.float)
x_coordinate, y_coordinate, z_coordinate = [], [], []
x_errors, y_errors, z_errors = [], [], []
overall_error = torch.tensor([0.], dtype=torch.float)
absolute_error = torch.tensor([0.], dtype=torch.float)
for _ in range(len(X_train)):
    predicted_output = model(X_train[_])
    x_coordinate.append(predicted_output[0])
    y_coordinate.append(predicted_output[1])
    z_coordinate.append(predicted_output[2])
    error_tensor += abs(y_train[_] - predicted_output)
    X_percentage_error += torch.mul(100, torch.div(abs(predicted_output[0] - y_train[_][0]), abs(y_train[_][0])))
    Y_percentage_error += torch.mul(100, torch.div(abs(predicted_output[1] - y_train[_][1]), abs(y_train[_][1])))
    Z_percentage_error += torch.mul(100, torch.div(abs(predicted_output[2] - y_train[_][2]), abs(y_train[_][2])))
    x_errors.append(abs(predicted_output[0] - y_train[_][0]))
    y_errors.append(abs(predicted_output[1] - y_train[_][1]))
    z_errors.append(abs(predicted_output[2] - y_train[_][2]))
    overall_error += abs(predicted_output[0] - y_train[_][0]) ** 2 + abs(predicted_output[1] - y_train[_][1]) ** 2 + abs(
            predicted_output[2] - y_train[_][2]) ** 2
    overall_error += abs(predicted_output[0] - y_train[_][0]) + abs(
        predicted_output[1] - y_train[_][1]) + abs(
        predicted_output[2] - y_train[_][2])
    # print(_)
X_outputs = torch.stack(x_coordinate)
Y_outputs = torch.stack(y_coordinate)
Z_outputs = torch.stack(z_coordinate)
error = error_tensor.div(len(X_train))
X_errors = torch.stack(x_errors)
Y_errors = torch.stack(y_errors)
Z_errors = torch.stack(z_errors)
X_errors = X_errors.cpu()
Y_errors = Y_errors.cpu()
Z_errors = Z_errors.cpu()
X_errors = X_errors.detach().numpy()
Y_errors = Y_errors.detach().numpy()
Z_errors = Z_errors.detach().numpy()
print("Total error Train set ==> ", error_tensor)
#print("Mean error of the Train set ==> ", error_tensor.div(len(X_train)))
print("Mean and Standard Deviation of X Coordinate ==> ", X_outputs.mean(), X_outputs.std())
print("Mean and Standard Deviation of Y Coordinate ==> ", Y_outputs.mean(), Y_outputs.std())
print("Mean and Standard Deviation of Z Coordinate ==> ", Z_outputs.mean(), Z_outputs.std())
print("X Percentage error of the Train set ==> ", X_percentage_error.div(len(X_train)))
print("Y Percentage error of the Train set ==> ", Y_percentage_error.div(len(X_train)))
print("Z Percentage error of the Train set ==> ", Z_percentage_error.div(len(X_train)))
print("Mean error of the Train Set X ==> "+str(X_errors.mean())+" Standard Deviation ==> "+str(X_errors.std()))
print("Mean error of the Train Set Y ==> "+str(Y_errors.mean())+" Standard Deviation ==> "+str(Y_errors.std()))
print("Mean error of the Train Set Z ==> "+str(Z_errors.mean())+" Standard Deviation ==> "+str(Z_errors.std()))
print("Overall error of a point of the Train set ==> ", torch.sqrt(overall_error.div(len(X_train))))
print("Absolute Mean error of a point of the Train set ==> ", absolute_error.div(len(X_train)))

X_outputs = X_outputs.cpu()
Y_outputs = Y_outputs.cpu()
Z_outputs = Z_outputs.cpu()
X_outputs = X_outputs.detach().numpy()
Y_outputs = Y_outputs.detach().numpy()
Z_outputs = Z_outputs.detach().numpy()


ax = sns.distplot(X_errors)
ax.set(xlabel='Normal Distribution M='+str(X_errors.mean())+' SD='+str(X_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Y_errors)
ax.set(xlabel='Normal Distribution M='+str(Y_errors.mean())+' SD='+str(Y_errors.std()), ylabel='Frequency')
plt.show()

ax = sns.distplot(Z_errors)
ax.set(xlabel='Normal Distribution M='+str(Z_errors.mean())+' SD='+str(Z_errors.std()), ylabel='Frequency')
plt.show()
#plt.hist(X_outputs, edgecolor='black')
#plt.show()
#plt.savefig(plotsave + 'test_1_x.png')

#plt.hist(Y_outputs, edgecolor='black')
#plt.show()
#plt2.savefig(plotsave + 'test_1_y.png')

#plt.hist(Z_outputs, edgecolor='black')
#plt.show()
#plt3.savefig(plotsave + 'test_1_z.png')







# number of parameters criteria
print("*********************************************")
print("The 2n+d number ==> ", 2 * len(X_train) + len(X_train[0]))
print("The actual number of parameters ==> ", D_in * H1 + H1 * H2 + H2 * D_out)





import pickle

file_train_loss = open(plotsave + 'Train_Loss', 'rb')
train_loss_values = pickle.load(file_train_loss)
file_train_loss.close()

file_validation1_loss = open(plotsave + 'Validation1_Loss', 'rb')
validation1_loss_values = pickle.load(file_validation1_loss)
file_validation1_loss.close()

file_validation2_loss = open(plotsave + 'Validation2_Loss', 'rb')
validation2_loss_values = pickle.load(file_validation2_loss)
file_validation2_loss.close()

file_validation3_loss = open(plotsave + 'Validation3_Loss', 'rb')
validation3_loss_values = pickle.load(file_validation3_loss)
file_validation3_loss.close()

import matplotlib.pyplot as plt

plt.plot(train_loss_values, label='Train Loss')
plt.plot(validation1_loss_values, label='Validation1 Loss')
plt.plot(validation2_loss_values, label='Validation2 Loss')
plt.plot(validation3_loss_values, label='Validation3 Loss')
plt.ylabel('Loss Values')
plt.xlabel('Iterations')

#plt.legend('Train Loss', 'Validation1 Loss', 'Validation2 Loss', 'Validation3 Loss')
labels = ['Train Loss', 'Validation1 Loss', 'Validation2 Loss', 'Validation3 Loss']
plt.legend(labels)
plt.xlim(0, 100)
plt.ylim(0, 200)

plt.show()
#plt10.savefig(plotsave + 'loss.png')




import pickle
import torch
import matplotlib.pyplot as plt
trainLoss_errors, validation1Loss_errors, validation2Loss_errors, validation3Loss_errors = [], [], [], []
file_trainLossError = open(plotsave+'Train_Loss', 'rb')
train_error = pickle.load(file_trainLossError)
file_trainLossError.close()
from scipy.stats import norm
import seaborn as sns

print("Train error values", train_error)
ax = sns.distplot(train_error)
ax.set(xlabel='Normal Distribution', ylabel='Frequency')
plt.show()
#file_validation1LossError = open(plotsave+'Validation1_Loss', rb)



