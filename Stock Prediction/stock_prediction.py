import csv
import numpy as np
from sklearn.svm import SVR #Support Vector Regression
import matplotlib.pyplot as plt


dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # to avoid indexes in the dataset
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[2]))
            prices.append(float(row[1]))
    return

def predict_price(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1



    svr_lin = SVR(kernel= 'linear', C=1e3) #Linear svr
    svr_poly = SVR(kernel= 'poly', C=1e2, degree=2) #polynomial svr
    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1) #Radial basis function

    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='red', label='Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color='green', label='Polynomial Model')
    plt.plot(dates, svr_rbf.predict(dates), color='blue', label='RBF Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('AAPL.csv')

predicted_price = predict_price(dates,prices,29)






