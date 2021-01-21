import pandas as pd
import numpy as np
from scipy.stats import ks_2samp as ks
from scipy.stats import chisquare, chi2_contingency, fisher_exact


def get_contingency(data1, data2, all_values, cat):
    # Store the values for contingency
    all_d = {}
    # Create contingency
    for i in cat:
        # Check all_values is list or pandas table
        if isinstance(all_values, list):
            lst = all_values
        else:
            lst = sorted(pd.unique(all_values[i].dropna()))

        arr = []
        for k in lst:
            a = data1[i].loc[data1[i] == k].count()
            b = data2[i].loc[data2[i] == k].count()
            arr.append([a, b])
        # Although the results won't different, we need to transpose the matrix.
        arr = np.transpose(arr).tolist()
        all_d[i] = arr
    return all_d


def chi2_cal(data1, data2, all_values, cat):
    """
    all_values: The dataframe which contains all of the multi categorical values.
    cat: An array or list which contains all of the features need to be calculated.
    """

    # Store the result of chi2
    result_d = {}

    # Create contingency
    all_d = get_contingency(data1, data2, all_values, cat)

    for key, value in all_d.items():
        # if chi2_contingency(value)[1] < 0.05:
        result_d[key] = {
            "statistic": round(chi2_contingency(value)[0], 2),
            "pvalue": round(chi2_contingency(value)[1], 4),
        }

    return result_d


def ks_cal(data1, data2, num):
    """
    num: the features of the data1 and data2
    """
    # Store the results
    ks_rs = {}
    for i in num:
        result = ks(data1[i].dropna(), data2[i].dropna(),
                    mode='auto', alternative='two-sided')
        if result[1] < 0.05:
            # data structure
            ks_rs[i] = {}
            ks_rs[i]['statistic'] = result[0]
            ks_rs[i]['pvalue'] = result[1]
            # print(i, result[0], result[1])
            greater = ks(data1[i].dropna(), data2[i].dropna(),
                         mode='auto', alternative='greater')
            less = ks(data1[i].dropna(), data2[i].dropna(),
                      mode='auto', alternative='less')
            if greater[1] < 0.05:
                ks_rs[i]['tail_greater'] = True
                ks_rs[i]['tail_pvalue_great'] = greater[1]
                # print('greater:', greater[1],)
            else:
                ks_rs[i]['tail_greater'] = False
                ks_rs[i]['tail_pvalue_great'] = None

            if less[1] < 0.05:
                ks_rs[i]['tail_less'] = True
                ks_rs[i]['tail_pvalue_less'] = less[1]
                # print('less:', less[1],)
            else:
                ks_rs[i]['tail_less'] = False
                ks_rs[i]['tail_pvalue_less'] = None
            # print('---------')
    return ks_rs


def chi2_count(data1, data2, all_values, cat, store={}):
    """
    all_values: The dataframe which contains all of the multi categorical values.
    cat: An array or list which contains all of the features need to be calculated.
    """
    # Create contingency
    all_d = get_contingency(data1, data2, all_values, cat)

    for key, value in all_d.items():
        if chi2_contingency(value)[1] < 0.05:
            store[key] = store.get(key, 0)+1

    return store


def ks_count(data1, data2, num, store={}):
    for i in num:
        result = ks(data1[i].dropna(), data2[i].dropna(), mode='auto',
                    alternative='two-sided')
        if result[1] < 0.05:

            store[i] = store.get(i, 0)+1

    return store


def ks_not_pass(data1, data2, num,):
    count = list()
    for i in num:
        result = ks(data1[i].dropna(), data2[i].dropna(), mode='auto',
                    alternative='two-sided')

        if result[1] < 0.05:
            count.append(i)

    return count


def chi2_not_pass(data1, data2, all_values, cat):
    """
    all_values: The dataframe which contains all of the multi categorical values.
    cat: An array or list which contains all of the features need to be calculated.
    """
    # Store the result of chi2

    count = list()
    # Create contingency
    all_d = get_contingency(data1, data2, all_values, cat)

    for key, value in all_d.items():
        if chi2_contingency(value)[1] < 0.05:
            count.append(key)
            # print(key, ":", chi2_contingency(value)[0], chi2_contingency(value)[1])
    return count


def NIHSS_constraint_check(data, column):
    # Check the sum of NIHSS
    NIHTotoal_check_pass = False
    if data.loc[data[column].sum(axis=1) > 42].shape[0] == 0:
        NIHTotoal_check_pass = True
    # Check the details of NIHSS
    # NIHSS 1a == 3 -> NIHSS XX == X
    condition = np.array(
        [None, 2, 2, None, None, 3, 4, 4, 4, 4, 0, 2, 3, 2, 2])
    loss = pd.DataFrame()
    for i, cond in enumerate(condition):
        if cond:

            loss =loss.append(data.loc[(data['NIHS_1a_in'] == 3) &
                             (data[column[i]] != cond)
                             ])

    return NIHTotoal_check_pass, loss.drop_duplicates()
