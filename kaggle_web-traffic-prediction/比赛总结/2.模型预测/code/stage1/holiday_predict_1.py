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
import re
import time

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


def holiday_model_log(df_train, lang, review=False):
    # data using to train
    df_train['y'] = df_train['y'].astype('float')
    df_train.y = np.log1p(df_train.y)
    # data to be predicted
    pre_df = pd.DataFrame(columns=['ds', 'yhat'], dtype='float64')
    pre_df['ds'] = pd.date_range('1/1/2017', '3/1/2017')

    if (isinstance(lang, float) and math.isnan(lang)):
        holidays = None
    else:
        holidays = holidays_dict[lang]

    try:
        m = Prophet(holidays=holidays)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=60)
        forecast = m.predict(future)

        pre_y = np.array(forecast['yhat'][-60:])
		pre_y = np.expm1(pre_y)
        pre_y[np.where(pre_y < 0)] = 0
        pre_df['yhat'] = np.array(pre_y)

    except RuntimeError:
        pre_df['yhat'] = 0
        return pre_df

    return pre_df


if __name__ == '__main__':
    start = time.time()
    # get page_date --> key
    key_df = pd.read_csv('../data/key_1.csv')
    # get train_data
    train = pd.read_csv("../data/train_1.csv")

    page_details = get_page_details(train)
    train_df = pd.concat([page_details, train], axis=1)

    # predict range
    pre_min = 130000
    pre_max = 140000
    # output
    output_df = pd.DataFrame(columns=['Id', 'Visits'])
    output_df['Id'] = key_df['Id'][60 * pre_min:60*pre_max]

    pages_id = []
    for i in range(pre_min, pre_max):
        page = re.sub('_\d+-\d+-\d+', '', key_df['Page'][60 * i])
        page_id = train[train['Page'] == page].index.values[0]
        pages_id.append(page_id)

    for index, page_row_id in enumerate(pages_id):
        print 'process row_num {0} record'.format(str(page_row_id))
        df_train = extract_series(train_df, page_row_id, 5)
        lang = train_df.iloc[page_row_id, 1]
        if not df_train['y'].any():
            df_train = df_train.fillna(0)
        prediction = holiday_model_log(df_train.copy(), lang, review=False)
        output_df['Visits'][60 * index:60 * (index + 1)] = np.array(prediction['yhat'])
    output_df.to_csv('../res/submission'+str(pre_max)+'.csv', index=False)
    print 'processing 10000 records elapse ', time.time() - start, ' seconds'
