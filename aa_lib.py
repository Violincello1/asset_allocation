from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import datetime
import seaborn as sns
import statsmodels
from arch import arch_model
from arch.univariate import GARCH
import pysolnp
import matplotlib as mpl
import random


class PortGroup:
    def plot_cumulative_returns(self):
        data = (1+self.returns_data).cumprod()
        return data.plot()

    def __init__(self, mr=None, ports=None, rf_rate=0):
        self.mean_variance = pd.DataFrame(
            index=ports, columns=['Annualized Returns', 'Standard Deviation', 'Sharpe'])
        self.mean_variance.loc[:, 'Annualized Returns'] = (
            1+mr.loc[:, ports]).prod()**(252/len(mr.index))-1
        self.mean_variance.loc[:, 'Standard Deviation'] = mr.loc[:, ports].std(
        )*np.sqrt(252)
        self.mean_variance.loc[:, 'Sharpe'] = \
            (self.mean_variance.loc[:, 'Annualized Returns'] -
             rf_rate)/self.mean_variance.loc[:, 'Standard Deviation']
        self.returns_data = mr.loc[:, ports]


class PortfolioGroups:
    def __init__(self, mr, all_ports):
        self.returns_data = mr
        self.mean_variance = pd.DataFrame(
            columns=['Annualized Returns', 'Standard Deviation', 'Sharpe', 'Risk Group'])
        self.ports = {}
        for ports in all_ports.keys():
            self.ports[ports] = PortGroup(mr=mr, ports=all_ports[ports])
            tmp = self.ports[ports].mean_variance
            tmp['Risk Group'] = ports
            self.mean_variance = pd.concat([self.mean_variance, tmp])


class PortfolioOptimization:
    def mean_variance(self, vol_range, start_asset_class=None):
        cov_matrix = self.cov_matrix.values
        expected_ret = self.expected_returns.values.flatten()
        num_asset_classes = cov_matrix.shape[0]
        starting_point = [0]*num_asset_classes
        if start_asset_class is None:
            starting_point[0] = 1
        else:
            starting_point[0] = start_asset_class
        inequality_lower_bounds = [0]*num_asset_classes + [vol_range[0]]
        inequality_upper_bounds = [.2]*num_asset_classes + [vol_range[1]]
        inequality_upper_bounds[0:4] = [1]*4

        def vol_function(weights):
            weights = np.atleast_2d(weights)
            vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
            return vol

        def sharpe(weights):
            weights = np.atleast_2d(weights)
            return -np.matmul(weights, expected_ret.T)/vol_function(weights)

        def eq_func(weights):
            return [sum(weights)]

        def in_eq_func(weights):
            return weights + [vol_function(weights)]

        def solve():
            result = pysolnp.solve(
                obj_func=sharpe,
                par_start_value=starting_point,
                ineq_func=in_eq_func,
                ineq_lower_bounds=inequality_lower_bounds,
                ineq_upper_bounds=inequality_upper_bounds,
                eq_func=eq_func,
                eq_values=[1])
            return result
        return solve()

    def __init__(self, asset_classes=[], returns=None, cov=None, expected_returns=None) -> None:
        self.asset_classes = asset_classes
        if cov is not None:
            self.cov_matrix = cov
        else:
            self.cov_matrix = returns.cov()
        if expected_returns is not None:
            self.expected_returns = expected_returns
        else:
            self.expected_returns = (1+returns).prod()**(1/returns.shape[0])-1


def get_ef(returns, cov=None, annualize=True):
    try:
        min_vol = returns.std().min().item()
        max_vol = returns.std().max().item()
    except:
        min_vol = returns.std().min()
        max_vol = returns.std().max()

    def ef_gen():
        vol = 1.0001*min_vol
        while vol <= 1.0001 * max_vol:
            yield PortfolioOptimization(returns=returns).mean_variance(vol_range=[0, vol]).solve_value, vol
            vol += (max_vol-min_vol)/100

    results = []
    for ret, vol in ef_gen():
        results.append([-ret, vol, -ret/vol])
    df = pd.DataFrame.from_records(
        results, columns=['Return', 'Volatility', 'Sharpe'])
    df.loc[:, 'Return'] = df.loc[:, 'Return'].cummax()
    if annualize:
        df.loc[:, 'Return'] = (1+df.loc[:, 'Return'])**252 - 1
        df.loc[:, 'Volatility'] *= np.sqrt(252)
    return df


def plot_ef(ef, ports, title='Efficient Frontier'):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    ax.plot(ef['Volatility'].values, ef['Return'].cummax().values)
    idx = ef['Sharpe'].idxmax()
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/ports.shape[0]) for i in range(ports.shape[0])]
    random.shuffle(colors)
    ax.set_prop_cycle(color=colors)
    legend_list = []
    for r in ports['Risk Group'].unique():
        df = ports.query('`Risk Group` == @r').sort_values('Volatility')
        x_max = [df.loc[df.index[0], 'Volatility']]
        y_max = [df.loc[df.index[0], 'Return']]
        x_min = [df.loc[df.index[-1], 'Volatility']]
        y_min = [df.loc[df.index[-1], 'Return']]
        for i in df.index[1:]:
            if df.loc[i, 'Return'] > y_max[-1]:
                x_max.append(df.loc[i, 'Volatility'])
                y_max.append(df.loc[i, 'Return'])
        for i in df.index[:-1][::-1]:
            if df.loc[i, 'Return'] < y_min[-1]:
                x_min.append(df.loc[i, 'Volatility'])
                y_min.append(df.loc[i, 'Return'])
        ax.fill(x_max+x_min, y_max+y_min, linewidth=3, alpha=.5, label=r)
        for i in df.index:
            ax.scatter(x=df.loc[i, 'Volatility'],
                       y=df.loc[i, 'Return'], label=i)

    ax.scatter(
        ef.loc[idx, 'Volatility'],
        ef.loc[idx, 'Return'],
        marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.title(title)
    plt.legend(labelspacing=0.8, ncol=5, bbox_to_anchor=(1, 0))
