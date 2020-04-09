import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from LSTM import LSTM
from ts_generator import generate_ar_k


def get_data():
    training_set = np.array([generate_ar_k(np.array([3.5, -4.85, 3.325, -1.1274, 0.1512]), N=1440, noise_func=np.random.normal)]).transpose()

    def sliding_windows(data, seq_length):
        x = []
        y = []
        for i in range(len(data) - seq_length - 1):
            _x = data[i:(i + seq_length)].squeeze()
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)
        return np.array(x), np.array(y)

    global sc
    sc = MinMaxScaler()
    # Scale the result to 0 to 1.
    training_data = sc.fit_transform(training_set)

    seq_length = 4
    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size
    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return trainX, testX, trainY, testY


# get the data
Xtrain, Xtest, Ytrain, Ytest = get_data()

# The model will be a sequence of layers (?)
model = LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)
loss = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.0611,momentum=0.9)


def train(model, loss, optimizer, inputs, labels):
    inputs = Variable(inputs)
    labels = Variable(labels)

    optimizer.zero_grad()

    # Forward
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)

    # Backward
    output.backward()
    optimizer.step()

    return output.item()

def predict(model, inputs):
    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits.data.numpy()


epochs = 2000
batch_size = 32
n_batches = Xtrain.size()[0] // batch_size

costs = []
test_accuracies = []
for i in range(epochs):
    cost = 0.
    for j in range(n_batches):
        Xbatch = Xtrain[j * batch_size:(j+1)*batch_size]
        Ybatch = Ytrain[j * batch_size:(j+1)*batch_size]
        cost += train(model, loss, optimizer, Xbatch, Ybatch)
    # Xbatch = Xtrain
    # Ybatch = Ytrain
    # cost += train(model, loss, optimizer, Xbatch, Ybatch)

    Ypred = predict(model, Xtest)
    print("Epoch: %d, cost(train): %.6f, cost(test): %.6f" % ((i + 1), cost, loss.forward(torch.from_numpy(Ypred), Ytest)))

    costs.append(cost)
# sc = MinMaxScaler()
# real_pred = sc.inverse_transform(Ypred)
# plt.plot(costs)
# plt.title('Training cost')
# plt.show()
TrainPred = predict(model, Xtrain)
plt.plot(sc.inverse_transform(Ytest), 'r')
plt.plot(sc.inverse_transform(Ypred))
plt.show()

plt.plot(sc.inverse_transform(Ytrain), 'r')
plt.plot(sc.inverse_transform(TrainPred))
plt.show()

