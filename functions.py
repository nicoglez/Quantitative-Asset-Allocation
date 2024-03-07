import pandas as pd
import yfinance as yf
import numpy as np
from typing import Optional
import datetime as dt


class asset_allocation:

    def __init__(self, data_stocks: pd.DataFrame, data_benchmark: pd.DataFrame, rf: Optional[float] = .05):
        self.stocks = data_stocks
        self.stocks_rends = data_stocks.pct_change().dropna()
        self.bench = data_benchmark
        self.bench_rends = data_benchmark.pct_change().dropna()
        self.rf = rf

    # Get metrics
    @staticmethod
    def get_metrics(w, mean, cov):
        returns = np.sum(mean * w) * 252
        var = np.dot(w, np.dot(cov, w)) ** 0.5 * 252
        std = np.dot(w, np.dot(cov, w)) ** 0.5 * (252 ** 0.5)
        return var, std, returns

    # Downside Risk
    @staticmethod
    def downside_risk(diff: pd.DataFrame):
        downside = diff[diff <= 0].fillna(0)
        std = downside.std()
        return np.array(std)

    # Upside Risk
    @staticmethod
    def upside_risk(diff: pd.DataFrame):
        upside = diff[diff >= 0].fillna(0)
        std = upside.std()
        return np.array(std)

    # Min Variance Optimization
    def min_var(self, n_port):
        mean = self.stocks_rends.mean()
        cov = self.stocks_rends.cov()
        n_stocks = len(self.stocks.columns)
        history = np.zeros((1, n_port))
        w = []
        # Montecarlo simulation
        for i in range(n_port):
            temp = np.random.uniform(0, 1, n_stocks)
            temp = temp / np.sum(temp)
            w.append(temp)
            var, std, r = self.get_metrics(temp, mean, cov)
            history[0, i] = var

        # return optimal weights
        return w[np.argmin(history)]

    # Sharpe Ratio optimization
    def sharpe_ratio(self, n_port):
        mean = self.stocks_rends.mean()
        cov = self.stocks_rends.cov()
        n_stocks = len(self.stocks.columns)
        history = np.zeros((1, n_port))
        w = []
        # Montecarlo simulation
        for i in range(n_port):
            temp = np.random.uniform(0, 1, n_stocks)
            temp = temp / np.sum(temp)
            w.append(temp)
            var, std, r = self.get_metrics(temp, mean, cov)
            history[0, i] = (r - self.rf) / std

        # return optimal weights
        return w[np.argmax(history)]

    # Semivariance Optimization
    def semivariance(self, n_port):
        # Calculate downside risk
	corr = self.stocks_rends.corr()
        diff = self.stocks_rends - self.bench_rends.values
        std = self.downside_risk(diff)
        # Calculate semivar matrix
        semivar_matrix = np.multiply(std.reshape(len(std), 1), std) * corr
        # Montecarlo simulation
        mean = self.stocks_rends.mean()
        cov = self.stocks_rends.cov()
        n_stocks = len(self.stocks.columns)
        history = np.zeros((1, n_port))
        w = []
        # Montecarlo simulation
        for i in range(n_port):
            temp = np.random.uniform(0, 1, n_stocks)
            temp = temp / np.sum(temp)
            w.append(temp)
            history[0, i] = np.dot(temp, np.dot(semivar_matrix, temp))

        return w[np.argmin(history)]

    # Omega Optimization
    def omega(self, n_port):
        # Calculate downside risk
        diff = self.stocks_rends - self.bench_rends.values
        downside = self.downside_risk(diff)
        upside = self.upside_risk(diff)
        omega = upside / downside

        # Montecarlo simulation
        mean = self.stocks_rends.mean()
        cov = self.stocks_rends.cov()
        n_stocks = len(self.stocks.columns)
        history = np.zeros((1, n_port))
        w = []
        # Montecarlo simulation
        for i in range(n_port):
            temp = np.random.uniform(0, 1, n_stocks)
            temp = temp / np.sum(temp)
            w.append(temp)
            history[0, i] = np.dot(omega, temp)

        return w[np.argmax(history)]

    # Make summary
    def summary(self, n_port):
        # Get weights with multiple asset allocation tecniques
        w_MV = np.array([self.min_var(n_port) for i in range(20)]).mean(axis=0)
        w_SR = np.array([self.sharpe_ratio(n_port) for i in range(20)]).mean(axis=0)
        w_SV = np.array([self.semivariance(n_port) for i in range(20)]).mean(axis=0)
        w_O = np.array([self.omega(n_port) for i in range(20)]).mean(axis=0)

        # Create df
        df = pd.DataFrame()
        df["Min Var"] = w_MV
        df["Max Sharpe"] = w_SR
        df["Semivariance"] = w_SV
        df["Omega"] = w_O
        df["Stocks"] = self.stocks.columns

        df.set_index("Stocks", inplace=True)

        return df.T


class backtesting:

    def __init__(self, weights_summary: pd.DataFrame, data_stocks: pd.DataFrame,
                 data_benchmark: pd.DataFrame, cap_inicial: int):
        self.weights = weights_summary
        self.returns = data_stocks.pct_change().dropna()
        self.bench = data_benchmark.pct_change().dropna()
        self.capital = cap_inicial

    @property
    def history(self):
        # Copy returns to dont mess with the initial variables
        returns = self.returns.copy()
        weights = self.weights.copy()
        # Change pandas display format
        pd.options.display.float_format = '{:,.4f}'.format
        # Empty DF
        h = pd.DataFrame()
        # Make backtesting for the different weights of strategies
        for i in range(len(weights)):
            # get weights
            temp = weights.iloc[i, :]
            # get returns of strategy
            port_returns = 1 + (returns * temp).sum(axis=1)
            # make cumprod
            port_returns[0] = self.capital
            port_returns = port_returns.cumprod()
            # fill DF
            h[weights.iloc[i, :].name] = port_returns
        # Backtesting for benchmark
        rb = self.bench.copy()
        benchmark = 1 + rb
        # make cumprod
        benchmark.iloc[0] = self.capital
        benchmark = benchmark.cumprod()
        # fill DF
        h["Benchmark"] = benchmark

        return h

    def metrics(self, rf: float):
        # get portfolio evolution
        evol = self.history.copy()
        # get metrics
        rets = evol.pct_change().dropna()
        returns = evol.pct_change().dropna().mean() * 252
        std = evol.pct_change().dropna().std() * 252 ** 0.5
        sharpe = (returns - rf) / std
        # create df
        m = pd.DataFrame()
        m["Annualized Return"] = returns
        m["Annualized Vol"] = std
        m["Sharpe Ratio"] = sharpe
        m['VaR 97.5%'] = np.percentile(rets, 100-97.5, axis=0)
        return m


class download_data:

    def __init__(self, start_date=str, end_date=str, benchmark: Optional[str] = None,
                 tickers_USA: Optional[str] = None, tickers_MX: Optional[str] = None):
        self.USA = tickers_USA
        self.MX = tickers_MX
        self.bench = benchmark
        self.start = start_date
        self.end = end_date

    def download(self) -> tuple:

        if self.USA:
            # download USA data
            closes = pd.DataFrame(yf.download(self.USA, start=self.start, end=self.end, progress=False)["Adj Close"])
            closes.reset_index(inplace=True)
            closes['Date'] = closes['Date'].dt.tz_localize(None)
            # download USD/MXN data
            closes_TC = pd.DataFrame(yf.download("MXN=X", start=self.start, end=self.end, progress=False)["Adj Close"])
            closes_TC.reset_index(inplace=True)
            closes_TC['Date'] = closes_TC['Date'].dt.tz_localize(None)
            closes_TC.set_index("Date", inplace=True)
            # join
            closes = closes.join(closes_TC, on="Date", how="left")
            # multiply exchange rate and prices
            final = pd.DataFrame()
            for i in closes.columns:
                if i == "Date":
                    final[i] = closes.Date

                elif i == "Adj Close":
                    pass

                else:
                    final[i] = closes[i] * closes["Adj Close"]

        if self.MX:
            # download MX stock data
            closes_mx = pd.DataFrame(yf.download(self.MX, start=self.start, end=self.end, progress=False)["Adj Close"])
            closes_mx.reset_index(inplace=True)
            closes_mx['Date'] = closes_mx['Date'].dt.tz_localize(None)
            closes_mx.set_index("Date", inplace=True)
            if self.USA:
                # join data
                final = final.join(closes_mx, on="Date", how="left")
                mx_tickers = closes_mx.columns

                for i_ticker in mx_tickers:
                    # fill na
                    v = []
                    for i in final[i_ticker][1:]:
                        if np.isnan(i):
                            try:
                                v.append(v[-1])
                            except:
                                v.append(final[i_ticker].dropna().iloc[0])

                        else:
                            v.append(i)

                    v = [v[0]] + v
                    # join filled data
                    final.drop(i_ticker, axis=1, inplace=True)
                    final[i_ticker] = v
            else:
                final = closes_mx

        final.set_index("Date", inplace=True)

        # Download data
        if self.bench:
            benchmark = pd.DataFrame(yf.download(self.bench, start=self.start, end=self.end, progress=False)["Adj Close"])
            benchmark.reset_index(inplace=True)
            benchmark['Date'] = benchmark['Date'].dt.tz_localize(None)
            benchmark.set_index("Date", inplace=True)

            return final, benchmark

        else:
            return final


class dynamic_backtesting:

    def __init__(self, data_stocks, data_opt, data_benchmark, data_opt_b, capital):
        self.data = data_stocks
        self.data_opt = data_opt
        self.bench = data_benchmark
        self.bench_opt = data_opt_b
        self.capital = capital

    # Split the df in n periods
    @staticmethod
    def split_df(data: pd.DataFrame, f_day: bool, yearly: bool):

        p_data = []
        unique_years = list(set([i.strftime("%Y") for i in data.index]))
        unique_years = [str(j) for j in sorted([int(i) for i in unique_years])]
        holds, typ = [], f_day
        counter = 0
        for i in unique_years:
            holds.append(dt.datetime.strptime(f"{i}-01-01", "%Y-%m-%d"))
            holds.append(dt.datetime.strptime(f"{i}-06-30", "%Y-%m-%d"))
        len_dates = len(holds)

        if yearly:
            for i in range(len_dates - 1):
                if i == (len_dates - 2):
                    p_data.append(data[(data.index > holds[i])])
                else:
                    p_data.append(data[(data.index >= holds[i]) & (data.index < holds[i + 2])])
        else:
            if not typ:
                holds = holds[1:]

            len_dates = len(holds)
            for i in range(len_dates):
                if i == (len_dates - 1):
                    p_data.append(data[(data.index > holds[i])])
                else:
                    p_data.append(data[(data.index > holds[i]) & (data.index < holds[i + 1])])

        return p_data

    # Metrics template
    @staticmethod
    def metrics(evol, rf):
        # get portfolio evolution
        evol = evol
        # get metrics
        rets = evol.pct_change().dropna()
        returns = evol.pct_change().dropna().mean() * 252
        std = evol.pct_change().dropna().std() * 252 ** 0.5
        sharpe = (returns - rf) / std
        # create df
        m = pd.DataFrame()
        m["Annualized Return"] = returns * 100
        m["Annualized Vol"] = std * 100
        m['VaR 97.5%'] = np.percentile(rets, 100-97.5, axis=0)*100

        return m

    # multiple asset allocation
    def multiple_AA(self, rf, sims):
        data_opt = self.data_opt.copy()
        data_benchmark_opt = self.bench_opt.copy()
        # split
        data = self.split_df(data=data_opt, f_day=True, yearly=True)[:-1]
        data_bm = self.split_df(data=data_benchmark_opt, f_day=True, yearly=True)[:-1]
        ponds = []
        # optimize
        for i in range(len(data)):
            ponds.append(asset_allocation(data[i], data_bm[i], rf).summary(sims))

        return ponds

    # make backtesting
    def backtesting(self, rf, sims, show_weights=False):
        # Copy data
        bt_data = self.split_df(data=self.data.copy(), f_day=True, yearly=False)
        data_benchmark = self.bench.copy()

        AA = self.multiple_AA(rf=rf, sims=sims)
        capital = self.capital


        # Start methods
        methods = ['Min Var', 'Max Sharpe', 'Semivariance', 'Omega', 'Benchmark']
        backtest = pd.DataFrame()
        total_fees, t_pl = {}, {}

        # Make dynamic backtesting for multiple methods
        for method in range(len(methods)):

            historical = [capital]
            fees, pl = [], []


            if methods[method] == "Benchmark":
                rb = data_benchmark.pct_change()
                benchmark = 1 + rb
                # make cumprod
                benchmark.iloc[0] = capital
                benchmark = benchmark.cumprod()
                backtest[methods[method]] = benchmark


            else:
                for sim in range(len(bt_data)):

                    n_stocks = np.floor((AA[sim].iloc[method, :] * historical[-1]) / bt_data[sim].iloc[0, :])
                    if sim == 0:
                        historical = np.sum(bt_data[sim] * n_stocks, axis=1)
                        fees.append(historical[0] * (1.16 * .00125))
                    else:
                        historical = pd.concat([historical, np.sum(bt_data[sim] * n_stocks, axis=1)])

                        new_position = bt_data[sim].iloc[0, :] * n_stocks
                        old_position = bt_data[sim - 1].iloc[0, :] * old_stocks
                        port_old_value = np.sum(old_position)
                        port_new_value = np.sum(new_position)
                        pl.append(port_new_value - port_old_value)
                        fees.append(sum(abs(new_position - old_position)) * ((1.16 * .00125)))

                    old_stocks = n_stocks

                t_pl.update({methods[method]: pl})
                total_fees.update({methods[method]: np.sum(fees)})
                historical[0] = capital
                backtest[methods[method]] = historical

        fees_df = pd.DataFrame(total_fees.items(), columns=["Method", "Charged Fees"])
        fees_df.set_index("Method", inplace=True)
        metrics = self.metrics(backtest, rf)

        if show_weights:
            for i in AA:
                try:
                    i["IUITN"]=i["Adj Close"]
                    i.drop("Adj Close", axis=1, inplace=True)
                except:
                    continue
            return backtest[["Semivariance", "Benchmark"]], metrics.T[["Semivariance", "Benchmark"]], AA[-1].T[["Semivariance"]]
        else:
            return backtest[["Semivariance", "Benchmark"]], metrics.T[["Semivariance", "Benchmark"]]

        
def get_weights(data, data_benchmark):
    
    if 'Adj Close' in data.columns:
        data["IUITN"]=data["Adj Close"]
        data.drop("Adj Close", axis=1, inplace=True)
    cl=asset_allocation(data[data.index>'2023-07-01'], data_benchmark[data_benchmark.index>'2023-07-01'])
    w=np.array([cl.semivariance(10000) for i in range(100)]).mean(axis=0)
    d=dict(zip(data.columns, w))
    
    df=pd.DataFrame(d, index=['weights']).T*100
    df.sort_values(by='weights', ascending=False, inplace=True)

    df['P2']=df['weights']*0.15
    df['P3']=df['weights']*0.45
    df['P4']=df['weights']*0.6
    df['P5']=df['weights']*0.75
    
    return df