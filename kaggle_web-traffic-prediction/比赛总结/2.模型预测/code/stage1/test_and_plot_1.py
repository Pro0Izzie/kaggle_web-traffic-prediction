# coding=utf8

'''
use holidays when modeling Prophet
all holidays strored in holidays_dict.pkl
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
from fbprophet import Prophet
import matplotlib as mpl
import pickle

# chinese character showing
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')

with open('../data/holidays_dict.pkl', 'rb') as holiday_file:
    holidays_dict = pickle.load(holiday_file)


def get_page_details(train_data):
    # get details about page,especially lang field
    page_details = train_data.Page.str.extract(r'(?P<topic>.*)\_(?P<lang>.*).'
                                               r'wikipedia.org\_(?P<access>.*)\_(?P<type>.*)')
    return page_details


def get_train_validate_set(train_df, test_percent):
    # get train and test dataset
    train_end = math.floor((train_df.shape[1] - 5) * (1 - test_percent))
    train_end = int(train_end)
    train_ds = train_df.iloc[:, np.r_[0, 1, 2, 3, 4, 5:train_end]]
    test_ds = train_df.iloc[:, np.r_[0, 1, 2, 3, 4, train_end:train_df.shape[1]]]

    return train_ds, test_ds


def extract_series(df, row_num, start_idx=5):
    # test some row_num page
    y = df.iloc[row_num, start_idx:]
    df = pd.DataFrame({'ds': y.index, 'y': y.values})
    return df


def smape(predict, actual, debug=False):
    '''
    predict and actual is a panda series.
    In this implementation I will skip all the datapoint with actual is null
    '''
    actual = actual.fillna(0)
    data = pd.concat([predict, actual], axis=1, keys=['predict', 'actual'])
    data = data[data.actual.notnull()]
    # print data
    if debug:
        print('debug', data)

    evals = abs(data.predict - data.actual) * 1.0 / (abs(data.predict) + abs(data.actual)) * 2
    evals[evals.isnull()] = 0
    # print(np.sum(evals), len(data), np.sum(evals) * 1.0 / len(data))

    result = np.sum(evals) / len(data)

    return result


def plot_prediction_and_actual(model, forecast, actual, xlim=None, ylim=None, figSize=None, title=None):
    fig, ax = plt.subplots(1, 1, figsize=figSize)
    ax.set_ylim(ylim)
    ax.plot(pd.to_datetime(actual.ds), actual.y, 'r.')
    model.plot(forecast, ax=ax)
    ax.set_title(title)
    plt.show()


def holiday_model_log(df_train, df_actual, lang, review=False):
    start_date = df_actual.ds.min()
    end_date = df_actual.ds.max()

    actual_series = df_actual.y.copy()
    actual_series.index = df_actual.ds

    df_train['y'] = df_train['y'].astype('float')
    df_train.y = np.log1p(df_train.y)

    df_actual['y'] = df_actual['y'].astype('float')
    df_actual.y = np.log1p(df_actual.y)

    if (isinstance(lang, float) and math.isnan(lang)):
        holidays = None
    else:
        holidays = holidays_dict[lang]
    try:
        m = Prophet(holidays=holidays)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=60, include_history=False)
        print 'future:', future
        forecast = m.predict(future)

        if (review):
            ymin = min(df_actual.y.min(), df_train.y.min()) - 2
            ymax = max(df_actual.y.max(), df_train.y.max()) + 2
            plot_prediction_and_actual(m, forecast, df_actual, ylim=[ymin, ymax], figSize=(12, 4),
                                       title='Holiday model in log')
        mask = (forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)
        forecast_series = np.expm1(forecast[mask].yhat)
        forecast_series.index = forecast[mask].ds
        forecast_series[forecast_series < 0] = 0
        # print 'forecast_series\n', forecast_series
    except RuntimeError:
        all_zero_series = pd.Series(index=pd.date_range(start_date, end_date))
        all_zero_series = all_zero_series.fillna(0)
        print all_zero_series
        return smape(all_zero_series, actual_series)

    return smape(forecast_series, actual_series)


if __name__ == '__main__':
    train = pd.read_csv("../data/train_1.csv")

    page_details = get_page_details(train)
    train_df = pd.concat([page_details, train], axis=1)
    X_train, y_train = get_train_validate_set(train_df, 0.1)
    df_train = extract_series(X_train, 0, 5)
    df_actual = extract_series(y_train, 0, 5)
    if not df_train['y'].any():
        df_train = df_train.fillna(0)
    print df_train
    lang = X_train.iloc[0, 1]
    score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)
    print("The SMAPE score is : %.5f" % score)
