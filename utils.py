import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

from ts_generator import generate_ar_k


def get_data(alpha_array, series_length, seq_length, sc):
    training_set = np.array([generate_ar_k(alpha_array, N=series_length, noise_func=np.random.normal)]).transpose()

    def sliding_windows(data, seq_length):
        x = []
        y = []
        for i in range(len(data) - seq_length - 1):
            _x = data[i:(i + seq_length)].squeeze()
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)
        return np.array(x), np.array(y)

    # Scale the result to 0 to 1.
    training_data = sc.fit_transform(training_set)

    x, y = sliding_windows(training_data, seq_length)

    global train_size
    train_size = int(len(y) * 0.67)

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return trainX, testX, trainY, testY, dataX, dataY


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
