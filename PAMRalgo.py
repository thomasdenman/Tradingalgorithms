# region imports
from AlgorithmImports import *


# endregion

class FocusedYellowSardine(QCAlgorithm):
    def __init__(self):
        self.symbols = ["SPY", "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
                        "DIS", "DD", "XOM", "GE", "GS", "HD", "IBM", "INTC", "JPM", "MCD",
                        "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UTX", "UNH", "VZ", "V", "WMT"]

        self.num = 21 * 12
        self.PAMR1 = PAMR()
        self.winitial = self.PAMR1.init_weights(self.symbols)
        self.wtold = self.winitial

    def get_history(self, symbol):
        prices = []
        dates = []
        for i in self.history:
            bar = i[symbol]
            prices.append(np.array(float(bar.Close)))
            dates.append(bar.EndTime)
        symbol.df = pd.DataFrame({'price': prices}, index=dates)
        symbol.df['return'] = symbol.df['price'].diff()
        symbol.df = symbol.df.dropna()

    def Initialize(self):
        self.SetStartDate(2019, 5, 23)  # Set Start Date
        self.SetEndDate(2021, 9, 17)
        self.SetCash(100000)  # Set Strategy Cash
        self.SetWarmup(10)
        self.SetBenchmark("SPY")
        for i in range(len(self.symbols)):
            equity = self.AddEquity(self.symbols[i], Resolution.Daily).Symbol

        self.history = self.History(self.num, Resolution.Daily)

        self.pricedict = {}

        for i in self.symbols:
            self.pricedict[i] = []

        self.currentreturns = []

    def OnData(self, data):

        for i in self.symbols:
            try:
                self.pricedict[i].append(float(data[i].Close))
            except:
                pass
        returns = pd.DataFrame.from_dict(self.pricedict)

        if self.IsWarmingUp:
            return
        returns = returns.diff().dropna()
        return_arr = returns.to_numpy()
        self.currentreturns_arr = return_arr[:][-1]
        x = np.array(self.currentreturns_arr)

        self.wtnew = self.PAMR1.step(x, self.wtold)

        for i in range(0, len(self.symbols)):
            self.SetHoldings(self.symbols[i], self.wtnew[i])

        self.wtold = self.wtnew


class PAMR:

    def __init__(self, epsilon=0.2, C=500, variant=0):
        self.epsilon = epsilon
        self.C = C
        self.variant = variant

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b):
        b = self.update(last_b, x, self.epsilon, self.C)
        return b

    def update(self, b, x, eps, C):
        x_mean = np.mean(x)
        le = max(0.0, np.dot(b, x) - eps)

        if self.variant == 0:
            lam = le / np.linalg.norm(x - x_mean) ** 2
        elif self.variant == 1:
            lam = min(C, le / np.linalg.norm(x - x_mean) ** 2)
        elif self.variant == 2:
            lam = le / (np.linalg.norm(x - x_mean) ** 2 + 0.5 / C)
        lam = min(10000, lam)

        b = b - lam * (x - x_mean)

        return b

