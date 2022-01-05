import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error

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
    ind_rnd_dev = random.gauss(mean, float(std))
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
    df = pd.read_csv(normPath, parse_dates=True, index_col=0, sep=';')
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
    #let initial alpha at the end of period 15 be 0.2 and beta 0.1
    print(df.shape)
    df.sort_index(inplace=True)
    decompose_result = seasonal_decompose(df['demand'],model='multiplicative')
    decompose_result.plot()
    plt.show()

    alpha = 0.3
    #single exponential smoothing
    df['HWSE1'] = SimpleExpSmoothing(df['demand']).fit(smoothing_level=alpha, optimized=False, use_brute=True).fittedvalues
    df[['demand', 'HWSE1']].plot(title='Holt Winter Single Exponential Smoothing Graph')
    plt.show()
    #double exponential smoothing of dataset (multiplicative and additive trend)
    df['HWSE2_ADD'] = ExponentialSmoothing(df['demand'], trend='add').fit().fittedvalues
    df['HWSE2_MUL'] = ExponentialSmoothing(df['demand'], trend='mul').fit().fittedvalues
    df[['demand', 'HWSE2_ADD', 'HWSE2_MUL']].plot(title='Holt Winter Double Exponential Smoothing Graph - Additive '
                                                        'and Multiplicative Trend')
    train_demand = df[:42]
    test_demand = df[42:]

    fitted_model = ExponentialSmoothing(train_demand['demand'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
    test_predictions = fitted_model.forecast(6)
    train_demand['demand'].plot(legend=True,label='Train')
    #test_demand['demand'].plot(legend=True, label='Test',figsize=(6,4))
    test_predictions.plot(legend=True,label='Prediction')
    plt.title('Train, Test, Prediction data points using Holt Winters Exponential Smoothing')
    plt.show()

    #evaluation
    print(f'Mean Absolute Error = {mean_absolute_error(test_demand["demand"], test_predictions)}')
    print(f'Mean Squared Error = {mean_squared_error(test_demand["demand"], test_predictions)}')
    test_df = test_demand.tail(6)
    comparison = pd.concat(test_df, test_predictions, keys=["true demand", "prediction"])
    print(comparison)
    # export results in csv
    export_predictions = fitted_model.forecast((12))
    export_predictions = round(pd.DataFrame(export_predictions), 0)
    try:
        export_predictions.to_csv('export_predictions.csv', index=True, sep=';')
    except:
        print("Error occured during csv export")
    print(round(export_predictions, 0))
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
