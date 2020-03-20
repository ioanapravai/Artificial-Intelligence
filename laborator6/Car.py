import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

file_path_prices = "C:\\ANUL2\\Semestrul2\\IA\\laborator6\\data\\prices.npy"
file_path_training_data = "C:\\ANUL2\\Semestrul2\\IA\\laborator6\\data\\training_data.npy"

training_data = np.load(file_path_training_data)
prices = np.load(file_path_prices)
training_data = training_data[1:]
prices = prices[1:]

def normalize(train, test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train)
    standard_train = scaler.transform(train)
    standard_test = scaler.transform(test)
    return standard_train, standard_test


num_samples_fold = len(training_data) // 3
training_data_1, prices_1 = training_data[:num_samples_fold], prices[:num_samples_fold]
training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold],\
                            prices[num_samples_fold: 2 * num_samples_fold]
training_data_3, prices_3 = training_data[2 * num_samples_fold:], prices[2 * num_samples_fold:]


train1, test1 = normalize(training_data_1, training_data_2)
train2, test2 = normalize(training_data_2, training_data_3)
train3, test3 = normalize(training_data_1, training_data_3)

linear_regression_model = LinearRegression()
linear_regression_model.fit(train1, prices_1)
pred1 = linear_regression_model.predict(test1)
mse_value1 = mean_squared_error(prices_1, pred1)
mae_value1 = mean_absolute_error(prices_1, pred1)


linear_regression_model.fit(train2, prices_2)
pred2 = linear_regression_model.predict(test2)
mse_value2 = mean_squared_error(prices_2, pred2)
mae_value2 = mean_absolute_error(prices_2, pred2)

linear_regression_model.fit(train3, prices_3)
pred3 = linear_regression_model.predict(test3)
mse_value3 = mean_squared_error(prices_3, pred3)
mae_value3 = mean_absolute_error(prices_3, pred3)

mean_mse_value = (mse_value1 + mse_value2 + mse_value3)/3
mean_mae_value = (mae_value1 + mae_value2 + mae_value3)/3

print(mean_mae_value, mean_mse_value)