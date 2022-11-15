import os.path
import urllib.request
import pandas as pd
import progressbar
import random

from sklearn.model_selection import train_test_split

class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def download_data(url, file_name):
    if not os.path.isfile(file_name):
        print(f"Downloading file from {url} and storing it as {file_name}")
        urllib.request.urlretrieve(url, file_name, MyProgressBar())
        print(f"Finished downloading file from {url} and storing it as {file_name}")
    else:
        print(f"{file_name} exists")


def calc_target_variable(df, column_name, new_column_name, top_nvalue=25):
    top_n_car_makes = df[column_name].value_counts()[:top_nvalue].index.tolist()

    assert len(top_n_car_makes) == top_nvalue

    def func_gt_classifier(row):
        if row[column_name] in top_n_car_makes:
            val = 1
        else:
            val = 0

        return val

    df[new_column_name] = df.apply(func_gt_classifier, axis=1)
    return df


def data_preprocessing(url):
    file_name = "corrupted_data.csv"
    target_column_name = "Top25Manufacturer"

    download_data(url, file_name)

    print("Reading dataframe now")
    df = pd.read_csv(file_name, low_memory=False)

    # split dataset into actual test and train dataset
    # extract data where make is  available
    print("extract corrupted and non corrupted data as test and train data")
    df_test = df[df['Make'].isnull()]
    df = df[~df['Make'].isnull()]

    print("Test set shape:", df_test.shape)
    print("Train set shape:", df.shape)

    # drop columns which have a majority of their values missing
    df = df.drop(['Meter Id', 'Marked Time', 'VIN'], axis=1)

    '''
     As we have a lot of data we can drop the empty rows, when we have less data we can fill the missing data with the mean values for numerical variables and 
     mode values for categorical variables

     df = df.fillna(df.mean())
    '''
    df = df.dropna()

    df = calc_target_variable(df, "Make", target_column_name)

    # drop columns with too many unique values as they dont show any common features and features whose values are
    # not the right representation of the original values
    df = df.drop(['Ticket number', 'Make', 'Latitude', 'Longitude'], axis=1)

    # Handling data imbalance
    print("Handling Data Imbalance now")
    min_rows = min(len(df.query(f"{target_column_name}==1")), len(df.query(f"{target_column_name}==0")))
    class_1_idx = random.sample(list(df.query(f'{target_column_name}==1').index), min_rows)
    class_0_idx = random.sample(list(df.query(f'{target_column_name}==0').index), min_rows)

    # Use indices to select data
    df_new = df.loc[class_1_idx + class_0_idx]

    print("shape after data balancing:", df_new.shape)

    target = df_new[target_column_name]
    data = df_new.drop([target_column_name], axis=1)

    # split into train and valid data
    print("splitting data into train and valid sets")
    data_train, data_valid, target_train, target_valid = train_test_split(
        data, target, test_size=0.3, random_state=2022)

    return data_train, data_valid, target_train, target_valid




