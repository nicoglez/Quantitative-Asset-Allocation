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
            closes_TC = pd.DataFrame(yf.download("MXN=X", start=start_date, end=end_date)["Adj Close"])
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
            closes_mx = pd.DataFrame(yf.download(self.MX, start=start_date, end=end_date)["Adj Close"])
            closes_mx.reset_index(inplace=True)
            closes_mx['Date'] = closes_mx['Date'].dt.tz_localize(None)
            closes_mx.set_index("Date", inplace=True)
            if self.USA:
                # join data
                final = final.join(closes_mx, on="Date", how="left")
                # fill na
                v = []
                for i in final["Adj Close"][1:]:
                    if np.isnan(i):
                        v.append(v[-1])

                    else:
                        v.append(i)
                v = [v[0]] + v
                # join filled data
                final.drop("Adj Close", axis=1, inplace=True)
                final[self.MX] = v
                final.set_index("Date", inplace=True)
            else:
                final = closes_mx

        # Download data
        benchmark = pd.DataFrame(yf.download(self.bench, start=self.start, end=self.end)["Adj Close"])
        benchmark.reset_index(inplace=True)
        benchmark['Date'] = benchmark['Date'].dt.tz_localize(None)
        benchmark.set_index("Date", inplace=True)

        return final, benchmark