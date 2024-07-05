import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
import pandas as pd
import json
from sklearn.metrics import accuracy_score, roc_auc_score
from make_input import make_BD_input

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear = torch.nn.Linear(hidden_channels, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)
        zT = zt[..., -1, :]
        pred_y = self.readout(zT)
        return F.softmax(pred_y, dim=-1)
    
class build_ncde():

    def __init__(self, input_size, n_out, params):
        self.input_size = input_size
        self.n_out = n_out
        self.epochs = params['epochs']
        self.n_hidden = params['hidden size']
        self.bs = params['batch size']
        self.lr = params['lr']

    def fit(self, train_dl, val_dl):
        model = NeuralCDE(input_channels=self.input_size, hidden_channels=self.n_hidden, output_channels=self.n_out)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            roc_epochs = []
            for batch in train_dl:
                batch_coeffs, batch_y = batch
                pred_y = model(batch_coeffs).squeeze(-1)
                loss = criterion(pred_y, batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            for batch in val_dl:
                batch_coeffs, batch_y = batch
                pred_y = model(batch_coeffs).squeeze(-1)
                pred_prob_val = list(pred_y[:,1].data.numpy())
                y_true_val = list(batch_y.data.numpy())
                valroc = roc_auc_score(y_true_val, pred_prob_val)
                roc_epochs.append(valroc)
            print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))
            print('Epoch: {}   Training ROC AUC: {}'.format(epoch, valroc))
        return roc_epochs


if __name__ == "__main__":
    variables = ['SampleTakenmonth', 'CA125']
    variables = ['CA125']
    dir_train = "../data/BD_train.csv"
    dir_test = "../data/BD_test.csv"
    dir_params = "../data/params_to_select.json"

    with open(dir_params) as data_file:
        args = json.load(data_file)
    
    df_train = pd.read_csv(dir_train)
    df_test = pd.read_csv(dir_test)
    train_data, train_times, train_targets = make_BD_input(df_train, variables)
    test_data, test_times, test_targets = make_BD_input(df_test, variables)

    input_size = len(variables)
    n_out = 2
    bs = args['models_grid']['model_0']['batch size']
    params = args['models_grid']['model_0']

    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_data)
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_targets)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs)

    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_data)
    test_dataset = torch.utils.data.TensorDataset(test_coeffs, test_targets)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_data.size(0))

    ncde_model = build_ncde(input_size, n_out, params=params)
    rocaucs = ncde_model.fit(train_dataloader, test_dataloader)
    
    """
    input_size = train_data.size(-1)
    model = NeuralCDE(input_channels=input_size, hidden_channels=8, output_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 50
    for epoch in range(num_epochs):
        roc_epochs = []
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for batch in test_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            pred_prob_val = list(pred_y[:,1].data.numpy())
            y_true_val = list(batch_y.data.numpy())
            valroc = roc_auc_score(y_true_val, pred_prob_val)
            roc_epochs.append(valroc)

        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))
        print('Epoch: {}   Training ROC AUC: {}'.format(epoch, valroc))
    """