import pandas as pd
import yfinance as yf
import numpy as np
from typing import Optional


class asset_allocation:

    def __init__(self, data_stocks: pd.DataFrame, data_benchmark: pd.DataFrame, rf: float):
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
        diff = self.stocks_rends - self.bench_rends.values
        std = self.downside_risk(diff)
        # Calculate semivar matrix
        semivar_matrix = np.multiply(std.reshape(len(std), 1), std) * diff.corr()
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
        returns = evol.pct_change().dropna().mean() * 252
        std = evol.pct_change().dropna().std() * 252 ** 0.5
        sharpe = (returns - rf) / std
        # create df
        m = pd.DataFrame()
        m["Annualized Return"] = returns
        m["Annualized Vol"] = std
        m["Sharpe Ratio"] = sharpe

        return m

class download_data:

    def __init__(self, benchmark: str, start_date=str, end_date=str,
                 tickers_USA: Optional[str] = None, tickers_MX: Optional[str] = None):
        self.USA = tickers_USA
        self.MX = tickers_MX
        self.bench = benchmark
        self.start = start_date
        self.end = end_date

    def download(self) -> tuple:

        final = pd.DataFrame()

        if self.USA:
            # download USA data
            closes = pd.DataFrame(yf.download(self.USA, start=self.start, end=self.end)["Adj Close"])
            closes.reset_index(inplace=True)
            closes['Date'] = closes['Date'].dt.tz_localize(None)
            # download USD/MXN data
            closes_TC = pd.DataFrame(yf.download("MXN=X", start=self.start, end=self.end)["Adj Close"])
            closes_TC.reset_index(inplace=True)
            closes_TC['Date'] = closes_TC['Date'].dt.tz_localize(None)
            closes_TC.set_index("Date", inplace=True)
            # join
            closes = closes.join(closes_TC, on="Date", how="left")
            # multiply exchange rate and prices
            for i in closes.columns:
                if i == "Date":
                    final[i] = closes.Date

                elif i == "Adj Close":
                    pass

                else:
                    final[i] = closes[i] * closes["Adj Close"]

        if self.MX:
            # download MX stock data
            closes_mx = pd.DataFrame(yf.download(self.MX, start=self.start, end=self.end)["Adj Close"])
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
                    final[i_ticker if i_ticker != "Adj Close" else self.MX] = v
            else:
                final = closes_mx

        final.set_index("Date", inplace=True)
        # Download data
        benchmark = pd.DataFrame(yf.download(self.bench, start=self.start, end=self.end)["Adj Close"])
        benchmark.reset_index(inplace=True)
        benchmark['Date'] = benchmark['Date'].dt.tz_localize(None)
        benchmark.set_index("Date", inplace=True)

        return final, benchmark