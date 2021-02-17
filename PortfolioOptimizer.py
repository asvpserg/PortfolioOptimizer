#This Program attempts to optimize a user's portfolio by using the efficient frontier
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Build the portfolio
assets = []
numAssets = int(input("Number of assets in portfolio: "))
weight_amt = 1 / numAssets
weights = np.array([])
count = 1

for i in range(0,numAssets):
    stock = input(str(count) + ") Ticker symbol: ")
    assets.append(stock)
    count = count + 1
    
    weights = np.append(weights, weight_amt)

#Get the stocks/portfolio starting date and ending date
startDate = '2013-01-01'
today = datetime.today().strftime('%Y-%m-%d')

#Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

for stock in assets:
	df[stock] = web.DataReader(stock, data_source = 'yahoo', start = startDate, end = today)['Adj Close']

#Chart the adjusted closing price history of the stocks in the portfolio
title = 'Portfolio Adj. Close Price History'
my_stocks = df

for c in my_stocks.columns.values:
	plt.plot(my_stocks[c], label = c)

plt.title(title)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Adj. Price USD ($)', fontsize = 18)
plt.legend(my_stocks.columns.values, loc = 'upper left')
plt.show()

#Create the daily simple return **Implement log returns eventually
returns = df.pct_change()
# logReturns = np.log(df[stock]/df[stock].shift(1))
# print(returns)

#Create the annualized covariance matrix
cov_matrix_annual = returns.cov() * 252
# print(cov_matrix_annual)

#Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
# print(port_variance)

#Calculate the portfolio volatility (standard deviation)
port_volatility = np.sqrt(port_variance)
# print(port_volatility)

#Calculate the annual portfolio return
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252
# print(portfolioSimpleAnnualReturn)

#Show the expected annual return, volatility(risk), and variance
percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'

print('Expected annual return: ' + percent_ret)
print('Annual volatility/risk: ' + percent_vols)
print('Annual variance: ' + percent_var)

#**************************
#*****MONTE CARLO SIM******
#**************************

np.random.randn
num_ports = 5000
all_weights = np.zeros((num_ports, len(df.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
	# print(stocks.columns)
	weights = np.array(np.random.random(numAssets))
	# print("Random Weights:")
	# print(weights)
	# print("Rebalance:")
	weights = weights / np.sum(weights)
	# print(weights)

	#SAVE WEIGHTS
	all_weights[ind, :] = weights

	#EXPECTED RETURN
	# print("Expected Portfolio Return:")
	# exp_ret=np.sum((log_ret.mean()*weights)*252)
	ret_arr[ind] = np.sum((returns.mean() * weights) * 252)
	# print(exp_ret)

	#EXPECTED VOLATILITY
	# print("Expected Volatility:")
	# exp_vol=np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
	vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
	# print(exp_vol)

	#SHARPE RATIO
	# print("Sharpe Ratio:")
	# SR=exp_ret/exp_vol
	sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]
	# print(SR)

#****************************************
#*****MONTE CARLO SIM RESULTS CHART******
#****************************************

plt.figure(figsize = (12, 8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap = 'plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(port_volatility, portfolioSimpleAnnualReturn, c = 'red', s = 50, edgecolors = 'black')
plt.show()

#****************************
#*****EFFICIENT FRONTIER*****
#****************************

def get_ret_vol_sr(weights):
	weights=np.array(weights)
	ret=np.sum(returns.mean()*weights)*252
	vol=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
	sr=ret/vol
	return np.array([ret,vol,sr])

def neg_sharpe(weights):
	return get_ret_vol_sr(weights)[2]*-1

def check_sum(weights):
	return np.sum(weights)-1 #Returns 0 if the sum of the weights is 1

cons=({'type':'eq','fun':check_sum})
# bounds=((0,1),(0,1),(0,1),(0,1))

bounds = ((0,1))

if numAssets == 1:
	bounds = ((0,1))
elif numAssets == 2:
	bounds = ((0,1),(0,1))
elif numAssets == 3:
	bounds = ((0,1),(0,1),(0,1))
elif numAssets == 4:
	bounds = ((0,1),(0,1),(0,1),(0,1))
elif numAssets == 5:
	bounds = ((0,1),(0,1),(0,1),(0,1),(0,1))
elif numAssets == 6:
	bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))			
elif numAssets == 7:
	bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
elif numAssets == 8:
	bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))	
elif numAssets == 9:
	bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
elif numAssets == 10:
	bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))											

optimal_results=minimize(neg_sharpe,weights,method='SLSQP',bounds=bounds,constraints=cons)

print(optimal_results)
# print(optimal_results.x)
get_ret_vol_sr(optimal_results.x)
# print(get_ret_vol_sr(optimal_results.x))

frontier_y=np.linspace(0,0.3,100)

def minimize_volatility(weights):
	return get_ret_vol_sr(weights)[1]

frontier_volatility=[]

for possible_return in frontier_y:
	cons=({'type':'eq','fun':check_sum},{'type':'eq','fun':lambda w: get_ret_vol_sr(w)[0]-possible_return})

	result=minimize(minimize_volatility,weights,method='SLSQP',bounds=bounds,constraints=cons)
	frontier_volatility.append(result['fun'])

plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
# plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')
plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)
plt.show()

#********************************
#*****PORTFOLIO OPTIMIZATION*****
#********************************

#Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
s = risk_models.sample_cov(df)

#Optimize for max sharpe ratio
ef = EfficientFrontier(mu, s)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)

#Get the discrete allocation of each share per stock
latest_prices = get_latest_prices(df)
weights = cleaned_weights
fa = int(input("Enter amount of investable funds: "))
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = fa)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))