import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import skew, kurtosis
from pandas.tseries.offsets import BDay




################################     Stocks selection and cointegretation     #######################################

def is_top_n(df_select, poids, n):
    contrib = df_select * poids
    top_n_stocks = contrib.apply(lambda x: x.sort_values(ascending=False).head(n).index, axis=1)
    stock_select = pd.DataFrame(False, index=df_select.index, columns=df_select.columns)
    for day, top_stocks in top_n_stocks.items():
        stock_select.loc[day, top_stocks] = True
    tickers = stock_select.sum(axis=0).sort_values(ascending=False).head(n).index.tolist()
    return sorted(tickers)


def select_stocks(T, n, freq_rank, stocks, poids, day_shift):
    if freq_rank==0:
        df_select = stocks[stocks.index <= T].tail(1)
    else:
        days = pd.date_range(start=T - BDay(int(252*freq_rank)), end=T, freq='B')[::-1][::day_shift][::-1]
        df_select = stocks.loc[days]
    tickers = is_top_n(df_select, poids, n)
    return tickers


def select_calibration_data(T, cal_period, tickers, indice, stocks):
    wind = 252
    #ret_ind = compute_ret(indice.loc[T-wind*BDay():T])
    #kurt = kurtosis(ret_ind, bias=True)
    #vol = ret_ind.std()
    period = int(252*cal_period)
    df_select = stocks[stocks.index <= T].tail(period)
    stocks_cal = df_select[tickers]
    indice_select = indice[indice.index <= T].tail(period)
    return stocks_cal, indice_select


def perform_lr_coint_test(stocks_cal, indice_select):
    X = sm.add_constant(np.log(stocks_cal))
    y = np.log(indice_select)
    lr = sm.OLS(y, X)
    model = lr.fit()
    params = model.params
    y_pred = model.predict(X)
    coint_t, _, critical_values = coint(y_pred, y)
    return params, coint_t, critical_values


#########################################   stats on index trackers  ###################################################

def compute_ret(portf):
    ret = np.log(portf).diff().dropna()
    return ret

def annual_ret(ret):
    ret_an = ret.groupby(ret.index.year).apply( lambda x: (np.exp(x.sum())-1) )
    return ret_an

def monthly_ret(ret):
    ret_month = ret.groupby(ret.index.to_period("M")).apply(lambda x: np.exp(x.sum()) - 1)
    ret_month.index = ret_month.index.to_timestamp()
    return ret_month

def annualized_ret(ret):
    return ret.mean()*252

def cumulative_ret(ret):
    return (ret.cumsum().apply(np.exp) - 1)

def annualized_vol(ret):
    return ret.std()*np.sqrt(252)

def excess_ret(ret, ret_ind):
    return ret - ret_ind

def correlation(ret, ret_ind):
    return ret.corr(ret_ind)

def ewma_correlation(ret, ret_ind, lambd=0.94):
    ewma_cov = ret.ewm(alpha=1-lambd).cov(ret_ind)
    ewma_vol1 = ret.ewm(alpha=1-lambd).std()
    ewma_vol2 = ret_ind.ewm(alpha=1-lambd).std()
    ewma_corr = ewma_cov / (ewma_vol1 * ewma_vol2)
    return ewma_corr

def ewma_volatility(ret, lambd =0.94):
    ewma_vol = np.sqrt(ret.pow(2).ewm(alpha=1 - lambd).mean())*np.sqrt(252)
    return ewma_vol


###################################     index tracking algorithm      ############################################

def transaction_cost(T, weigths_past, weigths, stocks):
    diff = weigths.sub(weigths_past, fill_value=0)
    P_T = stocks[diff.index].loc[T]
    #TC = 0.002*(abs(diff)*P_T).sum()
    TC = 0.002*abs(diff).sum()
    return TC


def tracking_portf(n, freq_rank, select_strat, cal_period, start_date, end_date, indice, day_shift, stocks, poids):
    if select_strat=='RD':
        select_shift = day_shift
    elif select_strat=='RSA':
        select_shift = int(252/2)
    elif select_strat=='RA':
        select_shift = 252
    
    dates_select = pd.date_range(start=start_date, end=end_date, freq=f"{select_shift}B")
    dates_rebalance = pd.date_range(start=start_date, end=end_date, freq=f"{day_shift}B")
    dates = pd.date_range(start=start_date, end=end_date, freq="1B")
    
    X_t, TC_t = np.zeros(len(dates)), np.zeros(len(dates))
    coint_t_list = []
    w_past = 0
    
    x0 = indice.loc[start_date]
    for i, t in enumerate(dates):
        
        if t in dates_select:
            tickers = select_stocks(t, n, freq_rank, stocks, poids, day_shift)
        
        if t in dates_rebalance:
            T = t
            stocks_cal, indice_select = select_calibration_data(T, cal_period, tickers, indice, stocks)
            params, coint_t, _ = perform_lr_coint_test(stocks_cal, indice_select)
            w = params.drop('const') / (params.drop('const').sum())
            coint_t_list.append(coint_t)
            
            TC = transaction_cost(T, w_past, w, stocks)
            TC_unit = TC/(min(day_shift, len(dates[i:])))
            P_T = stocks[tickers].loc[T]
            
            w_past = w
            X_T_1 = x0 if i==0 else X_t[i-1]

        P_t = stocks[tickers].loc[t]
        position = ((w/P_T)*P_t).sum()
        X_t[i] = X_T_1 * position
        TC_t[i] = TC_unit
        
    ret = compute_ret(pd.Series(X_t, index=dates))
    return ret, pd.Series(TC_t[1:], index=dates[1:]), pd.Series(coint_t_list, index=dates_rebalance)

"""
def results_tracking(trackers_dict, tracked_index, start_date):
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    res_tracking = []
    for (n, cal_period) in trackers_dict.keys():
        ret = trackers_dict[(n, cal_period)][0]
        TC_unit = trackers_dict[(n, cal_period)][1]
        adf = trackers_dict[(n, cal_period)][2].mean()
        ret_tc = ret - TC_unit
        
        exc_ret = excess_ret(ret, ret_ind)
        exc_ret_an = 100*annualized_ret(exc_ret)
        exc_vol_an = 100*annualized_vol(exc_ret)
        exc_ret_corr = correlation(exc_ret, ret_ind)
        ret_corr = correlation(ret, ret_ind)
        ret_sharpe = annualized_ret(ret)/annualized_vol(ret)
        ret_skew = skew(ret, bias=False)
        ret_kurt = kurtosis(ret, bias=False)
        
        exc_ret_tc = excess_ret(ret_tc, ret_ind)
        exc_ret_tc_an = 100*annualized_ret(exc_ret_tc)
        exc_vol_tc_an = annualized_vol(exc_ret_tc)
        exc_ret_tc_corr = correlation(exc_ret_tc, ret_ind)
        ret_tc_corr = correlation(ret_tc, ret_ind)
        ret_tc_sharpe = annualized_ret(ret_tc)/annualized_vol(ret_tc)
        ret_tc_skew = skew(ret_tc, bias=False)
        ret_tc_kurt = kurtosis(ret_tc, bias=False)
        
        res_tracking.append({'n':n, 'cal_period':cal_period, 'exc_ret_an(%)':exc_ret_an, 'exc_vol_an(%)':exc_vol_an, 'exc_ret_corr':exc_ret_corr, 
                            'ret_corr':ret_corr, 'sharpe':ret_sharpe, 'ret_skew':ret_skew, 'ret_kurt':ret_kurt, 'exc_ret_tc_an(%)':exc_ret_tc_an, 
                            'exc_vol_tc_an(%)':exc_vol_tc_an, 'exc_ret_tc_corr':exc_ret_tc_corr, 'ret_tc_corr':ret_tc_corr, 'sharpe_tc':ret_tc_sharpe,
                            'ret_tc_skew':ret_tc_skew, 'ret_tc_kurt':ret_tc_kurt, 'adf':adf})
    df_res_tracking = pd.DataFrame(res_tracking)
    return df_res_tracking
"""    

def index_features(indice, start_date):
    ret_ind = compute_ret(indice.loc[start_date:])
    ret_an = 100*annualized_ret(ret_ind)
    vol_an = 100*annualized_vol(ret_ind)
    ret_shar = ret_an / vol_an
    ret_skew = skew(ret_ind, bias=False)
    ret_kurt = kurtosis(ret_ind, bias=False)
    ind_features = pd.DataFrame([{'ret_an(%)':ret_an, 'vol_an(%)':vol_an, 'ret_sharpe':ret_shar, 'ret_skew':ret_skew, 'ret_kurt':ret_kurt}])
    return ind_features.T
    

def results_tracking(trackers_dict, tracked_index, start_date):
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    res_ret = []
    for (n, cal_period) in trackers_dict.keys():
        ret = trackers_dict[(n, cal_period)][0]
        TC_unit = trackers_dict[(n, cal_period)][1]
        adf = trackers_dict[(n, cal_period)][2].mean()
        ret_tc = ret - TC_unit
        
        ret_an = 100*annualized_ret(ret)
        vol_an = 100*annualized_vol(ret)
        ret_corr = correlation(ret, ret_ind)
        ret_shar = ret_an / vol_an
        ret_skew = skew(ret, bias=False)
        ret_kurt = kurtosis(ret, bias=False)
        
        ret_tc_an = 100*annualized_ret(ret_tc)
        vol_tc_an = 100*annualized_vol(ret_tc)
        ret_tc_corr = correlation(ret_tc, ret_ind)
        ret_tc_shar = ret_tc_an / vol_tc_an
        ret_tc_skew = skew(ret_tc, bias=False)
        ret_tc_kurt = kurtosis(ret_tc, bias=False)
        
        res_ret.append({'n':n, 'cal_period':cal_period, 'adf':adf, 'ret_an(%)':ret_an, 'vol_an(%)':vol_an, 'ret_sharpe':ret_shar, 
                             'ret_corr':ret_corr, 'ret_skew':ret_skew, 'ret_kurt':ret_kurt, 
                             'ret_tc_an(%)':ret_tc_an, 'vol_tc_an(%)':vol_tc_an, 'ret_tc_shar':ret_tc_shar, 
                             'ret_tc_corr':ret_tc_corr, 'ret_tc_skew':ret_tc_skew, 'ret_tc_kurt':ret_tc_kurt})
    df_res_ret = pd.DataFrame(res_ret)
    return df_res_ret

def results_exc_ret(trackers_dict, tracked_index, start_date):
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    res_exc_ret = []
    for (n, cal_period) in trackers_dict.keys():
        ret = trackers_dict[(n, cal_period)][0]
        TC_unit = trackers_dict[(n, cal_period)][1]
        adf = trackers_dict[(n, cal_period)][2].mean()
        ret_tc = ret - TC_unit
        exc_ret = excess_ret(ret, ret_ind)
        exc_ret_tc = excess_ret(ret_tc, ret_ind)
        
        exc_ret_an = 100*annualized_ret(exc_ret)
        exc_vol_an = 100*annualized_vol(exc_ret)
        exc_ret_corr = correlation(exc_ret, ret_ind)
        exc_ret_shar = exc_ret_an / exc_vol_an
        exc_ret_skew = skew(exc_ret, bias=False)
        exc_ret_kurt = kurtosis(exc_ret, bias=False)
        
        exc_ret_tc_an = 100*annualized_ret(exc_ret_tc)
        exc_vol_tc_an = 100*annualized_vol(exc_ret_tc)
        exc_ret_tc_corr = correlation(exc_ret_tc, ret_ind)
        exc_ret_tc_shar = exc_ret_tc_an / exc_vol_tc_an
        exc_ret_tc_skew = skew(exc_ret_tc, bias=False)
        exc_ret_tc_kurt = kurtosis(exc_ret_tc, bias=False)
        
        res_exc_ret.append({'n':n, 'cal_period':cal_period, 'adf':adf, 'exc_ret_an(%)':exc_ret_an, 'exc_vol_an(%)':exc_vol_an, 'exc_ret_sharpe':exc_ret_shar, 
                             'exc_ret_corr':exc_ret_corr, 'exc_ret_skew':exc_ret_skew, 'exc_ret_kurt':exc_ret_kurt, 
                             'exc_ret_tc_an(%)':exc_ret_tc_an, 'exc_vol_tc_an(%)':exc_vol_tc_an, 'exc_ret_tc_shar':exc_ret_tc_shar, 
                             'exc_ret_tc_corr':exc_ret_tc_corr, 'exc_ret_tc_skew':exc_ret_tc_skew, 'exc_ret_tc_kurt':exc_ret_tc_kurt})
    df_res_exc_ret = pd.DataFrame(res_exc_ret)
    return df_res_exc_ret
    




##########################################    Functions for plots    ########################################

def plot_cumulatives(n, cal_period, trackers_dict, tracked_index, start_date):
    trackers_n = {k: v for k, v in trackers_dict.items() if k[0] == n}
    trackers_cal = {k: v for k, v in trackers_dict.items() if k[1] == cal_period}
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    ind_cum_ret = 100*cumulative_ret(ret_ind)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.75))
    axs[0].set_title(f'n={n} stocks')
    axs[1].set_title(f'period={cal_period} yrs')
    axs[0].plot(ind_cum_ret, linewidth=0.5, label='index')
    axs[1].plot(ind_cum_ret, linewidth=0.5, label='index')
    for id in trackers_n:
        ret = trackers_n[id][0]
        cum_ret = 100*cumulative_ret(ret)
        axs[0].plot(cum_ret, linewidth=0.5, label=f'period={id[1]} yrs')    
        axs[0].set_ylabel('cum return %')
    for id in trackers_cal:
        ret = trackers_cal[id][0]
        cum_ret = 100*cumulative_ret(ret)
        axs[1].plot(cum_ret, linewidth=0.5, label=f'n={id[0]}stocks') 
        axs[1].set_ylabel('cum return %') 
    for i in range(2):
        for label in axs[i].get_xticklabels():
            label.set_rotation(45)  
    axs[0].legend() 
    axs[1].legend()
    plt.tight_layout()
   
    
def plot_ret_an(n, cal_period, trackers_dict, tracked_index, start_date):
    trackers_n = {k: v for k, v in trackers_dict.items() if k[0] == n}
    trackers_cal = {k: v for k, v in trackers_dict.items() if k[1] == cal_period}
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    ind_ret_an = 100*annual_ret(ret_ind)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
    axs[0].set_title(f'n={n} stocks')
    axs[1].set_title(f'period={cal_period} yrs')
    axs[0].plot(ind_ret_an, linewidth=1., label='index')
    axs[1].plot(ind_ret_an, linewidth=1., label='index')
    axs[0].set_ylabel('return %')
    axs[1].set_ylabel('reurn %')
    for id in trackers_n:
        ret = trackers_n[id][0]
        ret_an = 100*annual_ret(ret)
        axs[0].plot(ret_an, linewidth=1., label=f'period={id[1]} yrs')   
    for id in trackers_cal:
        ret = trackers_cal[id][0]
        ret_an = 100*annual_ret(ret)
        axs[1].plot(ret_an, linewidth=1., label=f'n={id[0]}stocks') 
    axs[0].legend() 
    axs[1].legend()
    plt.tight_layout()
    
    
def plot_ewma_vol_corr(trackers_dict, tracked_index, start_date, lambd=0.94):
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    ewma_vol_ind = 100*ewma_volatility(ret_ind, lambd)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.75))
    axs[0].set_title('EWMA volatility')
    axs[0].plot(ewma_vol_ind, linewidth=0.8, label='index')
    axs[0].set_ylabel('vol %')
    axs[1].set_title('EWMA correlation')
    for id in trackers_dict:
        ret = trackers_dict[id][0]
        ewma_vol = 100*ewma_volatility(ret, lambd)
        ewma_corr = ewma_correlation(ret, ret_ind, lambd)
        axs[0].plot(ewma_vol, linewidth=0.8, label=f'{id[1]}yrs, {id[0]}stocks')
        axs[1].plot(ewma_corr, linewidth=0.8, label=f'{id[1]}yrs, {id[0]}stocks')
    for i in range(2):
        for label in axs[i].get_xticklabels():
            label.set_rotation(45)
    axs[0].legend() 
    axs[1].legend()
    plt.tight_layout()
    
    
    
    
###################################     Long-Short Market neutral strategies  #################################

def build_index_plus(indice, x):  
    log_ret = np.log(indice).diff()
    c_daily = np.log(1+x) / 252 
    log_ret_plus = log_ret + c_daily
    log_ret_plus.iloc[0] = 0.0
    indice_plus = indice.iloc[0] * np.exp(log_ret_plus.cumsum())
    indice_plus = pd.Series(indice_plus, index=indice_plus.index)
    return indice_plus

def build_index_min(indice, x):  
    log_ret = np.log(indice).diff()
    c_daily = np.log(1+x) / 252 
    log_ret_minus = log_ret - c_daily
    log_ret_minus.iloc[0] = 0.0
    indice_minus = indice.iloc[0] * np.exp(log_ret_minus.cumsum())
    indice_minus = pd.Series(indice_minus, index=indice_minus.index)
    return indice_minus


def long_short_portf(n_plus, n_min, freq_rank, select_strat, cal_period, start_date, end_date, ind_plus, ind_min, day_shift, stocks, poids):
    tracker_plus_ret, tracker_plus_TC_unit, _ = tracking_portf(n_plus, freq_rank, select_strat, cal_period, start_date, end_date, 
                                                        ind_plus, day_shift, stocks, poids)
    tracker_min_ret, tracker_min_TC_unit, _ = tracking_portf(n_min, freq_rank, select_strat, cal_period, start_date, end_date, 
                                                        ind_min, day_shift, stocks, poids)
    ret_ls = tracker_plus_ret - tracker_min_ret
    ret_ls_tc = ret_ls - (tracker_plus_TC_unit + tracker_min_TC_unit)
    return ret_ls, ret_ls_tc


def plot_ret_ls_an(ls_dict):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    axs[0].set_title(f'Without fees')
    axs[1].set_title(f'With fees')
    axs[0].set_ylabel('return %')
    axs[1].set_ylabel('reurn %')
    for id in ls_dict:
        ret_ls = ls_dict[id][0]
        ret_ls_tc = ls_dict[id][1]
        axs[0].plot(100*annual_ret(ret_ls), linewidth=1., label=f'+{id[0]}/-{id[1]}')   
        axs[1].plot(100*annual_ret(ret_ls_tc), linewidth=1., label=f'+{id[0]}/-{id[1]}')   
    for i in range(2):
        for label in axs[i].get_xticklabels():
            label.set_rotation(45)
    axs[0].legend() 
    axs[1].legend()
    plt.tight_layout()


def plot_ret_month_ls(ls_dict):
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.75))
    axs[0].set_title(f'without transaction fees')
    axs[1].set_title(f'with transaction fees')
    for id in ls_dict:
        ls_ret = ls_dict[id][0]
        ls_ret_tc = ls_dict[id][1]
        ret_month = 100*monthly_ret(ls_ret)
        ret_month_tc = 100*monthly_ret(ls_ret_tc)
        axs[0].plot(ret_month, linewidth=1., label=f'+{id[0]}/-{id[1]}')   
        axs[1].plot(ret_month_tc, linewidth=1., label=f'+{id[0]}/-{id[1]}')   
    for i in range(2):
        for label in axs[i].get_xticklabels():
            label.set_rotation(45)
    axs[0].legend() 
    axs[1].legend()
    plt.tight_layout()
    
def plot_ewma_vol_corr_ls(ls_dict, tracked_index, start_date, lambd=0.98):
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    ewma_vol_ind = 100*ewma_volatility(ret_ind, lambd)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(2):
        axs[i,0].set_title('EWMA volatility')
        axs[i,0].plot(ewma_vol_ind, linewidth=0.8, label='index')
        axs[i,0].set_ylabel('vol %')
        axs[i,1].set_title('EWMA correlation')
        for id in ls_dict:
            ret = ls_dict[id][i]
            ewma_vol = 100*ewma_volatility(ret, lambd)
            ewma_corr = ewma_correlation(ret, ret_ind, lambd)
            axs[i,0].plot(ewma_vol, linewidth=0.8, label=f'+{id[0]}/-{id[1]}')
            axs[i,1].plot(ewma_corr.iloc[3:], linewidth=0.8, label=f'+{id[0]}/-{id[1]}')
            axs[i,1].set_ylim(-1, 1)
        axs[i,0].legend() 
        axs[i,1].legend()
    for i in range(2):
        for j in range(2):
            for label in axs[i,j].get_xticklabels():
                label.set_rotation(45)
    plt.tight_layout()
    
    
def ls_features(ls_dict, tracked_index, start_date):
    ret_ind = compute_ret(tracked_index.loc[start_date:])
    res_ret = []
    for (plus, minus) in ls_dict.keys():
        ret = ls_dict[(plus, minus)][0]
        ret_tc = ls_dict[(plus, minus)][1]
        
        ret_an = 100*annualized_ret(ret)
        vol_an = 100*annualized_vol(ret)
        ret_corr = correlation(ret, ret_ind)
        ret_shar = ret_an / vol_an
        ret_skew = skew(ret, bias=False)
        ret_kurt = kurtosis(ret, bias=False)
        
        ret_tc_an = 100*annualized_ret(ret_tc)
        vol_tc_an = 100*annualized_vol(ret_tc)
        ret_tc_corr = correlation(ret_tc, ret_ind)
        ret_tc_shar = ret_tc_an / vol_tc_an
        ret_tc_skew = skew(ret_tc, bias=False)
        ret_tc_kurt = kurtosis(ret_tc, bias=False)
        
        res_ret.append({'+(%)':plus, '-(%)':minus, 'ret_an(%)':ret_an, 'vol_an(%)':vol_an, 'ret_sharpe':ret_shar, 
                             'ret_corr':ret_corr, 'ret_skew':ret_skew, 'ret_kurt':ret_kurt, 
                             'ret_tc_an(%)':ret_tc_an, 'vol_tc_an(%)':vol_tc_an, 'ret_tc_sharpe':ret_tc_shar, 
                             'ret_tc_corr':ret_tc_corr, 'ret_tc_skew':ret_tc_skew, 'ret_tc_kurt':ret_tc_kurt})
    df_res_ret = pd.DataFrame(res_ret)
    return df_res_ret

"""
def tracking_portf(n_plus, n_min, freq_rank, select_strat, cal_period, start_date, end_date, ind_plus, ind_min, day_shift, stocks, poids):
    ######    True function
    if select_strat=='RD':
        select_shift = day_shift
    elif select_strat=='RSA':
        select_shift = int(252/2)
    elif select_strat=='RA':
        select_shift = 252
    
    dates_select = pd.date_range(start=start_date, end=end_date, freq=f"{select_shift}B")
    dates_rebalance = pd.date_range(start=start_date, end=end_date, freq=f"{day_shift}B")
    dates = pd.date_range(start=start_date, end=end_date, freq="1B")
    
    X_plus_t, X_min_t = np.zeros(len(dates)), np.zeros(len(dates))
    TC_t = np.zeros(len(dates))
    w_past = 0.
    
    x0_plus = ind_plus.loc[start_date]
    x0_min = ind_min.loc[start_date]
    for i, t in enumerate(dates):
        
        if t in dates_select:
            tickers_plus = select_stocks(t, n_plus, freq_rank, stocks, poids, day_shift)
            tickers_min = select_stocks(t, n_min, freq_rank, stocks, poids, day_shift)
        
        if t in dates_rebalance:
            T = t
            stocks_cal_plus, indice_select_plus = select_calibration_data(T, cal_period, tickers_plus, ind_plus, stocks)
            stocks_cal_min, indice_select_min = select_calibration_data(T, cal_period, tickers_min, ind_min, stocks)
            
            params_plus, _, _ = perform_lr_coint_test(stocks_cal_plus, indice_select_plus)
            params_min, _, _ = perform_lr_coint_test(stocks_cal_min, indice_select_min)
            w_plus = params_plus.drop('const') / (params_plus.drop('const').sum())
            w_min = params_min.drop('const') / (params_min.drop('const').sum())
            w = w_plus.sub(w_min, fill_value=0)

            TC = transaction_cost(T, w_past, w, stocks)
            TC_unit = TC/(min(day_shift, len(dates[i:])))
            
            P_plus_T = stocks[tickers_plus].loc[T]
            P_min_T = stocks[tickers_min].loc[T]
            
            w_past = w
            X_plus_T_1 = x0_plus if i==0 else X_plus_t[i-1]
            X_min_T_1 = x0_min if i==0 else X_min_t[i-1]

        P_plus_t = stocks[tickers_plus].loc[t]
        pos_plus = ((w/P_plus_T)*P_plus_t).sum()
        X_plus_t[i] = X_plus_T_1 * pos_plus
        
        P_min_t = stocks[tickers_min].loc[t]
        pos_min = ((w/P_min_T)*P_min_t).sum()
        X_min_t[i] = X_min_T_1 * pos_min
        
        TC_t[i] = TC_unit
        
    ret_plus =  compute_ret(pd.Series(X_plus_t, index=dates))
    ret_min =  compute_ret(pd.Series(X_min_t, index=dates))
    ret = ret_plus - ret_min
    return ret, pd.Series(TC_t[1:], index=dates[1:])
    
    
    
    
def tracking_portf(n, freq_rank, select_strat, cal_period, start_date, end_date, indice, day_shift, stocks, poids):
    if select_strat=='RD':
        select_shift = day_shift
    elif select_strat=='RSA':
        select_shift = int(252/2)
    elif select_strat=='RA':
        select_shift = 252
    
    dates_select = pd.date_range(start=start_date, end=end_date, freq=f"{select_shift}B")
    dates_rebalance = pd.date_range(start=start_date, end=end_date, freq=f"{day_shift}B")
    dates = pd.date_range(start=start_date, end=end_date, freq="1B")
    
    X_t, Xc_t, TC_t = np.zeros(len(dates)), np.zeros(len(dates)), np.zeros(len(dates))
    coint_t_list = []
    w_past = 0
    
    x0 = indice.loc[start_date]
    for i, t in enumerate(dates):
        
        if t in dates_select:
            tickers = select_stocks(t, n, freq_rank, stocks, poids, day_shift)
        
        if t in dates_rebalance:
            T = t
            stocks_cal, indice_select = select_calibration_data(T, cal_period, tickers, indice, stocks)
            params, coint_t, _ = perform_lr_coint_test(stocks_cal, indice_select)
            w = params.drop('const') / (params.drop('const').sum())
            w = w / w.sum()
            coint_t_list.append(coint_t)
            
            TC = transaction_cost(T, w_past, w, stocks)
            P_T = stocks[tickers].loc[T]
            TC_unit = TC/(min(day_shift, len(dates[i:])))
            
            w_past = w
            X_T_1 = x0 if i==0 else X_t[i-1]
            Xc_T_1 = x0 if i==0 else Xc_t[i-1]

        P_t = stocks[tickers].loc[t]
        position = ((w/P_T)*P_t).sum()
        X_t[i] = X_T_1 * position
        Xc_t[i] = Xc_T_1 * (position - TC_unit)
        TC_t[i] = Xc_T_1*TC_unit 
        
    return pd.Series(X_t, index=dates), pd.Series(Xc_t, index=dates), pd.Series(TC_t, index=dates), pd.Series(coint_t_list, index=dates_rebalance)

def results_tracking(trackers_dict, tracked_index):
    ret_ind = compute_ret(tracked_index)
    res_tracking = []
    for (n, cal_period) in trackers_dict.keys():
        ret = compute_ret(trackers_dict[(n, cal_period)][0])
        exc_ret = excess_ret(ret, ret_ind)
        exc_ret_an = 100*annualized_ret(exc_ret)
        exc_vol_an = annualized_vol(ret)
        exc_ret_corr = correlation(exc_ret, ret_ind)
        ret_corr = correlation(ret, ret_ind)
        ret_skew = skew(ret, bias=False)
        ret_kurt = kurtosis(ret, bias=False)
        
        ret_tc = compute_ret(trackers_dict[(n, cal_period)][1])
        exc_ret_tc = excess_ret(ret_tc, ret_ind)
        exc_ret_tc_an = 100*annualized_ret(exc_ret_tc)
        exc_vol_tc_an = annualized_vol(ret_tc)
        exc_ret_tc_corr = correlation(exc_ret_tc, ret_ind)
        ret_tc_corr = correlation(ret_tc, ret_ind)
        ret_tc_skew = skew(ret_tc, bias=False)
        ret_tc_kurt = kurtosis(ret_tc, bias=False)
        
        adf = trackers_dict[(n, cal_period)][3].mean()
        
        res_tracking.append({'n':n, 'cal_period':cal_period, 'exc_ret_an':exc_ret_an, 'exc_vol_an':exc_vol_an, 'exc_ret_corr':exc_ret_corr, 
                            'ret_corr':ret_corr, 'ret_skew':ret_skew, 'ret_kurt':ret_kurt, 'exc_ret_tc_an':exc_ret_tc_an, 
                            'exc_vol_tc_an':exc_vol_tc_an, 'exc_ret_tc_corr':exc_ret_tc_corr, 'ret_tc_corr':ret_tc_corr, 
                            'ret_tc_skew':ret_tc_skew, 'ret_tc_kurt':ret_tc_kurt, 'adf':adf})
    df_res_tracking = pd.DataFrame(res_tracking)
    return df_res_tracking


"""