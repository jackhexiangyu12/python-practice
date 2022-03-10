# LinearRegression 是一个用于线性回归的机器学习库
from sklearn.linear_model import LinearRegression

# pandas 和 numpy 用于数据操作
import pandas as pd
import numpy as np

# matplotlib 和 seaborn 用于绘制图形
import matplotlib.pyplot as plt
# matplotlib inline
plt.style.use('seaborn-darkgrid')

# yahoo Finance用于获取数据
import yfinance as yf

Df = yf.download('GLD', '2008-01-01', '2020-6-22', auto_adjust=True)

Df = Df[['Close']]

Df = Df.dropna()

Df.Close.plot(figsize=(10, 7), color='r')
plt.ylabel("Gold ETF Prices")
plt.title("Gold ETF Price Series")
plt.show()

Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S_3', 'S_9']]

y = Df['next_day_price']

t = .8
t = int(t * len(Df))

X_train = X[:t]
y_train = y[:t]

X_test = X[t:]
y_test = y[t:]

Y = m1 * X1 + m2 * X2 + C
Gold ETF price = m1 * 3 days moving average + m2 * 15 days moving average + c



linear = LinearRegression().fit(X_train, y_train)
print("Linear Regression model")
print("Gold ETF Price (y) = %.2f * 3 Days Moving Average (x1) \
+ %.2f * 9 Days Moving Average (x2) \
+ %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))

predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 7))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold ETF Price")
plt.show()

r2_score = linear.score(X[t:], y[t:])*100
float("{0:.2f}".format(r2_score))

gold = pd.DataFrame()

gold['price'] = Df[t:]['Close']
gold['predicted_price_next_day'] = predicted_price
gold['actual_price_next_day'] = y_test
gold['gold_returns'] = gold['price'].pct_change().shift(-1)

gold['signal'] = np.where(gold.predicted_price_next_day.shift(1) < gold.predicted_price_next_day, 1, 0)

gold['strategy_returns'] = gold.signal * gold['gold_returns']
((gold['strategy_returns'] + 1).cumprod()).plot(figsize=(10, 7), color='g')
plt.ylabel('Cumulative Returns')
plt.show()

sharpe = gold['strategy_returns'].mean()/gold['strategy_returns'].std()*(252**0.5)
'Sharpe Ratio %.2f' % (sharpe)

import datetime as dt

current_date = dt.datetime.now()

data = yf.download('GLD', '2008-06-01', current_date, auto_adjust=True)
data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()

data['predicted_gold_price'] = linear.predict(data[['S_3', 'S_9']])
data['signal'] = np.where(data.predicted_gold_price.shift(1) < data.predicted_gold_price, "Buy", "No Position")

data.tail(1)[['signal', 'predicted_gold_price']].T

