"""This file is enabling .csv data files to be accessed as a path in an numerical way.
It has a DataSender class, which takes domain of the data as (hourly, daily etc) and index of the .csv file
And returns the path needed for Pandas.read_csv object as two forms:
url_test, url_train
The Class used for that is 'DataSender'"""
from pandas import read_csv
url_train = '/Users/mac/Desktop/Datasets/M4_Dataset/M4_train_set'  # General position of train files
url_test = '/Users/mac/Desktop/Datasets/M4_Dataset/M4_test_set'  # General position of test files

class PathManager:
    """sender = DataSender('daily', 3)\n
    test_data_path, train_data_path = sender.send()"""
    def __init__(self, data_name, index, colab=0):
        self.name = str(data_name)
        self.index = index
        self.colab = colab

    def send(self):
        if self.colab == 0:
            if self.name == 'daily':
                return url_test + f'/Daily_Test/daily_test_{self.index}.csv', \
                       url_train + f'/Daily_Train/daily_train_{self.index}.csv'
            if self.name == 'hourly':
                return url_test + f'/Hourly_Test/hourly_test_{self.index}.csv', \
                       url_train + f'/Hourly_Train/hourly_train_{self.index}.csv'
            if self.name == 'monthly':
                return url_test + f'/Monthly_Test/monthly_test_{self.index}.csv', \
                       url_train + f'/Monthly_Train/monthly_train_{self.index}.csv'
            if self.name == 'quarterly':
                return url_test + f'/Quarterly_Test/quarterly_test_{self.index}.csv', \
                       url_train + f'/Quarterly_Train/quarterly_train_{self.index}.csv'
            if self.name == 'weekly':
                return url_test + f'/Weekly_Test/weekly_test_{self.index}.csv', \
                       url_train + f'/Weekly_Train/weekly_train_{self.index}.csv'
            if self.name == 'yearly':
                return url_test + f'/Yearly_Test/yearly_test_{self.index}.csv', \
                       url_train + f'/Yearly_Train/yearly_train_{self.index}.csv'
        elif self.colab == 1:
            if self.name == 'daily':
                return url_test + f'/daily/daily_test_{self.index}.csv', \
                       url_train + f'/daily/daily_train_{self.index}.csv'
            if self.name == 'hourly':
                return url_test + f'/hourly/hourly_test_{self.index}.csv', \
                       url_train + f'/hourly/hourly_train_{self.index}.csv'


class PathToValues:
    """From the path information, we are getting the actual data inside .csv files as numpy arrays\n
    converter = StringToNumpyArray(train_path, test_path)\n
    test_data = converter.convert_train()
    train_data = converter.convert_test()"""
    def __init__(self, test_path, train_path):
        self.train_path = train_path
        self.test_path = test_path

    def convert_train(self):
        train_data = read_csv(self.train_path, sep=',', index_col=False)
        train_data = train_data.dropna()
        train_data = train_data.drop(columns='ds')
        train_data = train_data.values

        return train_data

    def convert_test(self):
        test_data = read_csv(self.test_path, sep=',', index_col=False)
        test_data = test_data.dropna()
        test_data = test_data.drop(columns='ds')
        test_data = test_data.values

        return test_data
