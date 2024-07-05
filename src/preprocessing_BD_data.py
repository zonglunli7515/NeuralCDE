import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
np.random.seed(41)

def scaling(df, col_names):
    scaler = StandardScaler()
    df[col_names] = scaler.fit_transform(df[col_names])
    return df

data = pd.read_excel("../data/alldata.xls")
IDs = list(data.VolunteerID.unique())
df = data.copy()
columns = ['Case/Control', 'VolunteerID', 'SampleTakenmonth',
    'CA125', 'Glycodelin', 'HE4', 
    'MSLN25', 'MMP745', 'CYFRA55']
columns_to_scale = ['CA125', 'Glycodelin', 'HE4', 
    'MSLN25', 'MMP745', 'CYFRA55']

train_test_split = True
noise = False

times = []
for idx in IDs:
    times += list(df.loc[df.VolunteerID == idx]['SampleTakenmonth'].values - df.loc[df.VolunteerID == idx]['SampleTakenmonth'].values[0])
df['SampleTakenmonth'] = times

for idx in IDs:
    if df.loc[df.VolunteerID == idx].shape[0] < 3:
       df.drop(df[df.VolunteerID == idx].index, inplace=True)

if noise == True:
    stds = df[columns_to_scale].std().tolist()
    for i, variable in enumerate(columns_to_scale):
        std = stds[i]
        noise = np.random.normal(0, std, df.shape[0])
        df[variable] += noise

scaling(df, columns_to_scale)

if train_test_split == True:

    IDs_new = list(df.VolunteerID.unique())
    train_size = int(len(IDs_new)/2)+1
    train_idx = list(np.random.choice(IDs_new, train_size, 
    replace = False))
    test_idx = list(set(IDs_new) - set(train_idx)) 

    df_train = df[df.VolunteerID == train_idx[0]]
    for idx in train_idx:
        if idx != train_idx[0]:
            this_df = df[df.VolunteerID == idx]
            df_train = df_train.append(this_df)
    df_train.reset_index(drop=True, inplace=True)

    df_test = df[df.VolunteerID == test_idx[0]]
    for idx in test_idx:
        if idx != test_idx[0]:
            this_df = df[df.VolunteerID == idx]
            df_test = df_test.append(this_df)
    df_test.reset_index(drop=True, inplace=True)

    df_train = df_train[columns]
    df_test = df_test[columns]

    df_train.to_csv('../data/BD_train.csv')
    df_test.to_csv('../data/BD_test.csv')

else:
    df = df[columns]
    df.to_csv('../data/BD_train.csv')