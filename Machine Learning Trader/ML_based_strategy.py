

from RTLearner import RTLearner

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from util import get_data
from marketsim import compute_portvals
from BagLearner import BagLearner

from matplotlib.pyplot import axvline

# function to calculate RSI
def RSI(prices , num_days, return_rs=False):
    
    # add in date range...
    rev = pd.Series([x for x in reversed(prices)])
    
    rev_range = range(0 , len(rev) - num_days )
    
    temp = [x for x in reversed(prices)]
    changes = (pd.Series(temp) - pd.Series(temp[1:])).tolist()

    
    ranges = [changes[ix+1:ix+num_days+1] for ix in rev_range]
    

    ups = [[x for x in sublist if x > 0] for sublist in ranges]
    downs = [[x for x in sublist if x < 0] for sublist in ranges]
    
    avg_gains = [sum(x) / num_days for x in ups]
    avg_losses = [sum(x) / num_days for x in downs]
    
    
    RS = [x / abs(y) for x,y in zip(avg_gains , avg_losses)]
    
    if return_rs:
        return RS
    
    RSI = [100 - 100 / (1 + x) for x in RS]
    
    RSI = [x for x in reversed(RSI)]
    
    return RSI


def ml_action(x):
    
    if x in to_buy:
        return "BUY"
    elif x in to_sell:
        return "SELL"
    else:
        return "DO NOTHING"
    
    
training_dates = pd.date_range('2005-01-01','2010-12-31')

train = get_data(['IBM'] , training_dates)
train['ten_days_out'] = train.IBM.pct_change(-10) * -1


# get 10-day rate of change
train['ROC_10_Days'] =  train.IBM.pct_change(10)

# get exponentially moving average
train['EMA_30_Day'] = pd.Series(pd.ewma(train['IBM'], span = 30, min_periods = 29))
train['EMA_50_Day'] = pd.Series(pd.ewma(train['IBM'], span = 50, min_periods = 49))
train['EMA_200_Day'] = pd.Series(pd.ewma(train['IBM'], span = 200, min_periods = 199))

train['RSI_14_Days'] = [np.nan]*14 + RSI(train.IBM , 14)

train['EMA_30_Price_Ratio'] = train['IBM'] / train['EMA_30_Day']
train['EMA_50_Price_Ratio'] = train['IBM'] / train['EMA_50_Day']
train['EMA_200_Price_Ratio'] = train['IBM'] / train['EMA_200_Day']

train["SPY_RSI_14_Days"] = [np.nan]*14 + RSI(train.SPY , 14)

  
train = train[train.index.isin(pd.date_range('2006-01-01','2009-12-31'))]


# get only fields needed to run machine learning algorithm
data_train_x = np.asarray(train[["EMA_30_Price_Ratio","EMA_200_Price_Ratio","SPY_RSI_14_Days"]])
data_train_y = np.asarray(train.ten_days_out.tolist())


# run decision tree algorithm on IBM's price data
#learner = RTLearner(leaf_size = 50, verbose = False) # constructor
#learner.addEvidence(data_train_x, data_train_y) # training step
#Y = learner.query(data_train_x) # query

# run bag learner algorithm on IBM's data
bag_learner = BagLearner(learner = RTLearner , kwargs = {"leaf_size":50} , bags = 15, 
                         boost = False, verbose = False)
bag_learner.addEvidence(data_train_x , data_train_y)
Y = bag_learner.query(data_train_x)

Y = np.asarray(Y)

to_buy = [x for x in np.where(Y >= .01)[0]]
to_sell = [x for x in np.where(Y <= -.01)[0]]

to_buy = [train.index[x] for x in to_buy]
to_sell = [train.index[x] for x in to_sell]

# add field showing ML predictions
train['ML_Prediction'] = train.index.map(ml_action)


ix = 10
keep = [0]
last_action = train.iloc[0]['ML_Prediction']
while ix < train.shape[0]:

    if train.iloc[ix]['ML_Prediction'] + last_action in ['BUYSELL','SELLBUY','BUYDO NOTHING','SELLDO NOTHING']:
        keep.append(ix)
        last_action = ''.join([x for x in train.iloc[ix]['ML_Prediction']])
        ix = ix + 10

    else:
        ix = ix + 1



reset_train = train.reset_index()
reduced = reset_train[reset_train.index.isin(keep)]



reduced['Symbol'] = 'IBM'
orders = reduced[["index","Symbol","ML_Prediction"]]
orders.columns = ["Date","Symbol","Order"]


current_shares = orders.Order.map(lambda x: 500 if x == "BUY" else -500)
prior_shares = current_shares.shift(1).fillna(0)

diff_shares = current_shares - prior_shares

orders['Shares'] = abs(diff_shares)
orders = orders[orders.Shares != 0]

last_day = train[train.index == train.index[-1]].reset_index()

if diff_shares.sum() == 500:
    last_action = "SELL"
else:
    last_action = "BUY"

last_day = last_day[["index"]]
last_day.columns = ["Date"]
last_day["Symbol"] = "IBM"
last_day["Order"] = last_action
last_day["Shares"] = 500

orders = orders.append(last_day).reset_index(drop = True)

# write orders to csv
orders.to_csv("Machine_Learning_based_orders.csv",index = None)

print(orders)

# compute portfolio values
ml_based_values = compute_portvals("Machine_Learning_based_orders.csv",100000)



train['ML_Based_Value'] = ml_based_values
train['Buy_Hold_Value'] = train.IBM * 500 + (100000 - train.IBM[0] *  500)

train['Norm_ML_Based_Value'] = train.ML_Based_Value / train.ML_Based_Value[0]
train['Norm_Benchmark'] = train.Buy_Hold_Value / train.Buy_Hold_Value[0]

buys = orders[orders.Order == "BUY"].Date
sells = orders[orders.Order == "SELL"].Date

# generate performance plot
df_ML_Based_Value = train[["Norm_Benchmark","Norm_ML_Based_Value"]]
plt.figure() 
df_ML_Based_Value.plot(color = ['k','b']) 
plt.legend(loc=2,prop={'size':8})
plt.title("ML Prediction Performance versus Buy / Hold")

#max_range = [train.Norm_Rule_Based_Value.max() , train.Norm_Benchmark.max()]
plt.vlines(x=buys.tolist(), ymin=0.8, ymax=2.4, color ='g')
plt.vlines(x=sells.tolist(), ymin=0.8, ymax=2.4, color ='r')
plt.vlines(x=orders.Date.tolist(), ymin=0.8, ymax=2.4, color ='k',linestyle='dashed')

plt.savefig("ML Trader Performance vs. Benchmark.png")




