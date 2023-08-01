import pandas as pd
import yfinance as yf
import numpy as np
from typing import Optional


class download_data:

    def __init__(self, benchmark: str, start_date=str, end_date=str,
                 tickers_USA: Optional[str] = None, tickers_MX: Optional[str] = None):
        self.USA = tickers_USA
        self.MX = tickers_MX
        self.bench = benchmark
        self.start = start_date
        self.end = end_date

    def download(self) -> pd.DataFrame:

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