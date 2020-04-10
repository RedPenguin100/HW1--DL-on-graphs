import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import get_data, train, predict

sc = MinMaxScaler()
seq_length = 4
alpha_array = np.array([0.9])

Xtrain, Xtest, Ytrain, Ytest, dataX, dataY = get_data(alpha_array, series_length=1440, seq_length=seq_length, sc=sc)

# The model will be a sequence of layers (?)
model = torch.nn.Sequential()
model.add_module('dense1', torch.nn.Linear(seq_length, 8))
model.add_module('relu1', torch.nn.Sigmoid())
model.add_module('dense2', torch.nn.Linear(8, 1))

loss = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.0611, momentum=0.9)


epochs = 200  # Increase amount of epochs for better accuracy.
batch_size = 32
n_batches = Xtrain.size()[0] // batch_size

costs = []
test_accuracies = []
for i in range(epochs):
    cost = 0.
    for j in range(n_batches):
        Xbatch = Xtrain[j * batch_size:(j + 1) * batch_size]
        Ybatch = Ytrain[j * batch_size:(j + 1) * batch_size]
        cost += train(model, loss, optimizer, Xbatch, Ybatch)

    Ypred = predict(model, Xtest)
    print("Epoch: %d, cost(train): %.6f, cost(test): %.6f" % (
    (i + 1), cost, loss.forward(torch.from_numpy(Ypred), Ytest)))

    costs.append(cost)

TrainPred = predict(model, Xtrain)

# We will plot 2 graphs.

# The training data in red, and the trained model in blue
plt.plot(sc.inverse_transform(Ytrain), 'r')
plt.plot(sc.inverse_transform(TrainPred))
plt.show()

# The test data in red, and the trained model in blue
plt.plot(sc.inverse_transform(Ytest), 'r')
plt.plot(sc.inverse_transform(Ypred))
plt.show()

arr = Ytrain  # TODO: change to Ydata.
n = len(arr)
up_sum = 0.
down_sum = 0.
for k in range(1, n):
    up_sum += arr[k] * arr[k - 1]
    down_sum += arr[k - 1] ** 2

print("Estimated alpha value for the AR(1) series: {}".format(up_sum / down_sum))
