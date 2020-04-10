import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from LSTM import LSTM
from utils import get_data, train, predict
from ts_generator import get_best_prediction

sc = MinMaxScaler()

seq_length = 5
alpha_array = np.array([3.5, -4.85, 3.325, -1.1274, 0.1512])

Xtrain, Xtest, Ytrain, Ytest, dataX, dataY = get_data(alpha_array, series_length=1440, seq_length=seq_length, sc=sc)

train_size = len(Ytrain)
test_size = len(Ytest)

model = LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1, seq_length=seq_length)
loss = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 20  # Increase the epoch amount for
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

# Get values of best prediction:
entire_data = Ytrain.clone().squeeze()
for i in range(test_size):
    value = get_best_prediction(dataY[train_size - seq_length + i:train_size + i], alphas=alpha_array)
    entire_data = torch.cat((entire_data, torch.from_numpy(np.array([value])).float()), 0)

best_possible_prediction = entire_data[-test_size:].view(474, 1)

# Graph in red is the test model, graph in blue is the predicted model.
# Graph in green is the best possible predictor.
plt.plot(sc.inverse_transform(Ypred))
plt.plot(sc.inverse_transform(Ytest), 'r')
plt.plot(sc.inverse_transform(best_possible_prediction), 'g')
plt.show()

print("LSTM prediction MSE loss: {}".format(loss.forward(Ytest, torch.from_numpy(Ypred)).item()))
print("Best possible prediction MSE loss: {}".format(loss.forward(Ytest, best_possible_prediction).item()))
# As we may see, the best possible prediction is better than the LSTM model.
