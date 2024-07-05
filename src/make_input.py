import pandas as pd
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def make_BD_input(df, variables):
    IDs = list(df.VolunteerID.unique())
    ts_data = []
    ts_targets = []
    ts_times = []  
    for idx in IDs:
        ts_this = df.loc[df.VolunteerID==idx]
        df_this = torch.tensor(df.loc[df.VolunteerID==idx][variables].values, dtype=torch.float32)
        t_points = torch.tensor(ts_this.SampleTakenmonth.values, dtype=torch.float32)
        target = list(df.loc[df.VolunteerID==idx]['Case/Control'])[0]
        ts_data.append(df_this)
        ts_times.append(t_points)
        ts_targets.append(target)
    max_length = max([array.size(0) for array in ts_data])
    padded_train = []
    for array in ts_data:
        padded_train.append(fill_forward(array, max_length))
    padded_train = torch.stack(padded_train)
    ts_targets = torch.tensor(ts_targets, dtype=torch.int64)

    return padded_train, ts_times, ts_targets

def fill_forward(x, max_length):
    return torch.cat([x, x[-1].unsqueeze(0).expand(max_length - x.size(0), x.size(1))])


def process_time_interval(sample_times):
    time_intervals = []
    for times in sample_times:
        time_interval = torch.diff(times)
        time_intervals.append(torch.cat((torch.zeros(1), time_interval)))
    return time_intervals

if __name__ == "__main__":
    variables = ['CA125', 'Glycodelin', 'HE4', 'MSLN25', 'MMP745', 'CYFRA55']
    variables = ['SampleTakenmonth', 'CA125', 'Glycodelin', 'HE4', 'MSLN25', 'MMP745', 'CYFRA55']
    variables = ['SampleTakenmonth', 'CA125']
    dir_train = "../data/BD_train.csv"
    dir_test = "../data/BD_test.csv"
    df_train = pd.read_csv(dir_train)
    df_test = pd.read_csv(dir_test)
    train_data, train_times, train_targets = make_BD_input(df_train, variables)
    #test_data, test_times, test_targets = make_BD_input(df_test, variables)
    #train_data, train_times, train_targets = make_BD_input_interp(df_train, variables)
    