import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def constant_model():
    """
    From Sven Axsäter: Inventory Control, 2015
    Axsäter describes the Constant model as the simplest possible model. It's based on the assumption, that the demand
    in a certain period is represented by independent random deviations from an relatively stable average.
    Further he describes this as an easy and for many situations suitable approach. Especially the demand of
    everyday products like toothpaste, standard tools or spare parts can be estimated with the constant model.
    """
    mean = 0
    avg, demand_lst = cm_avg_data()
    std = cm_std_data(demand_lst)
    ind_rnd_dev = random.gauss(mean, std)
    plot_dec = input("Want to plot the Gaussian Graph? (y/n): ")
    if plot_dec == "y":
        plot_gauss(mean, std)
        return avg + ind_rnd_dev
    else:
        return avg + ind_rnd_dev


def plot_gauss(mu, sigma):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma))
    plt.show()
    return


def cm_std_data(demand_lst):
    std = np.std(demand_lst)
    print("The standard deviation is", np.around(std, 2))
    return std


def read_csv():
    path = "C:/Users/Lucas/wh_data.csv"
    normPath = path.replace(os.sep, '/')
    df = pd.read_csv(normPath, index_col='period')
    return df


def cm_avg_data():
    df = read_csv()
    demand_lst = df['demand'].tolist()
    avg = sum(demand_lst) / len(demand_lst)
    print("The average demand over", len(demand_lst), "periods is ", round(avg, 0))
    return avg, demand_lst


def expo_smoothing_model():
    """
    Forecasting with a regression model is an alternative technique to the exponential smoothing with trend described
    by Axsäter in section 2.5. The here implemented method takes the least square regression of demand/forecast errors.
    """
    df = read_csv()
    demand_lst = df['demand'].tolist()
    #forecast_lst = dataset['forecast'].tolist()
    # let initial alpha at the end of period 15 be 0.2 and beta 0.1
    alpha = 1 / (2 * len(demand_lst))
    print(df.shape())
    return


def forecast_demand():
    print("Which Forecasting Model you want to use?\n")
    print("1. Constant Model")
    print("2. Exponential Smoothing with Trend\n")
    chosen_model = input("Please enter 1 or 2 and confirm with Enter: \n")
    if chosen_model == "1":
        forecast = constant_model()
        print("\nThe forecast for the next period is:", round(forecast, 0))
    else:
        expo_smoothing_model()


if __name__ == "__main__":
    forecast_demand()
