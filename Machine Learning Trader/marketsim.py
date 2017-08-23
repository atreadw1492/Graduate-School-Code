

# load packages
import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
from pandas.io.data import get_data_yahoo


def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):

    # read in orders
    orders = pd.read_csv(orders_file)
    
    # convert Date field to pandas timestamp
    orders["Date"] = orders.Date.map(lambda x: pd.Timestamp(x))

    # set the index of orders to the date range
    orders = orders.set_index(orders.Date)
    
    # get list of unique symbols from orders data frame
    syms = orders.Symbol.unique().tolist()

    # get stock prices for each day in question for orders
    #stock_prices = get_data(syms , orders.Date).drop_duplicates()
    dates = pd.date_range(orders.index[0] , orders.index[-1])
    #stock_prices = get_data(syms , dates).drop_duplicates()
    stock_prices = get_data_yahoo(syms[0] , adjust_price = True)
    stock_prices = stock_prices[stock_prices.index.isin(dates)]

    stock_prices[syms[0]] = stock_prices.Close
    
    # merge orders data with stock prices
    #data = orders.join(stock_prices)
    data = stock_prices.join(orders)
 
    # initialize current portfolio value to the start value
    cash = start_val

    rows = range(0 , data.shape[0]) 
    cash_values = [cash]
    equities = [0]
    stock_shares = {x : 0 for x in stock_prices.columns} # initialize zero shares for each stock
    values = {x : 0 for x in stock_prices.columns}   
    for ix in rows:
        shares = data.Shares.iloc[ix] # number of shares in order ~ NaN if no order placed
        stock = data.Symbol.iloc[ix] # current stock in question
        
        # create a temp placeholder copy for stock_shares and for values
        # use these to handle leverage situations
        temp_shares = stock_shares.copy()
        temp_values = values.copy()

        if stock == stock:
            price = data[stock].iloc[ix]
            change = 1.0 * price * shares
        else:
            price = change = 0

        cash = cash_values[-1]
        temp_cash = float(np.copy(cash))
        #temp_cash = cash.copy() # for leverage
        order = data.Order.iloc[ix]
        
        #leverage = (sum(abs(all stock positions))) / (sum(all stock positions) + cash)

        # update shares accordingly
#==============================================================================
#         if shares == shares and order == "BUY":
#             stock_shares[stock] = stock_shares[stock] + shares
#         elif shares == shares and order == "SELL":
#             stock_shares[stock] = stock_shares[stock] - shares
#         else:
#             pass
#==============================================================================

        if shares == shares and order == "BUY":
            temp_shares[stock] = stock_shares[stock] + shares
        elif shares == shares and order == "SELL":
            temp_shares[stock] = stock_shares[stock] - shares
        else:
            pass


        
        # update equity values
#==============================================================================
#         for stock in stock_shares.keys():
#             values[stock] = stock_shares[stock] * data[stock].iloc[ix]
#==============================================================================
        
        if data.Order.iloc[ix] == "BUY":
            temp_cash = temp_cash - change
        elif data.Order.iloc[ix] == "SELL":
            temp_cash = temp_cash + change
        else:  
            pass
        
        for stock in stock_shares.keys():
            temp_values[stock] = temp_shares[stock] * data[stock].iloc[ix]
            #values[stock] = stock_shares[stock] * data[stock].iloc[ix]
            
        # don't place order if leverage is exceeded
            
        leverage = sum(map(abs,temp_values.values())) / (sum(map(abs,temp_values.values())) + temp_cash)

        if leverage < 3.0:
            cash = float(np.copy(temp_cash))
            values = temp_values.copy()
            stock_shares = temp_shares.copy()
        
        else:   # else leverage is exceeded, so don't perform order  
            print(ix)                     
            for stock in stock_shares.keys():
                #cash = cash + 0 # cash stays same
                # stock_shares remains same...
                # just need to update values
                for stock in stock_shares.keys():
                    values[stock] = stock_shares[stock] * data[stock].iloc[ix]
            
                pass                
                
                
        # update cash values
#==============================================================================
#         if data.Order.iloc[ix] == "BUY":
#             cash = cash - change
#         elif data.Order.iloc[ix] == "SELL":
#             cash = cash + change
#         else:  
#             pass
#==============================================================================
        
        cash_values.append(cash)
        equities.append(sum(values.values()))
    
    cash_values = cash_values[1:]
    equities = equities[1:]
    
    data["Cash"] = cash_values
    data['Equities'] = equities
    data['Total_Value'] = data['Cash'] + data['Equities']
    
    temp = data.copy()
    del temp['Symbol'] , temp['Order'] , temp['Shares'] , temp['Cash'] , temp['Equities']
    
#==============================================================================
#     dups = data.index.duplicated()
#     np.where(dups == True)
#==============================================================================
#==============================================================================
#     sp = get_data(["SPY"] , temp.index)
#     temp.join(sp)
#==============================================================================
#==============================================================================
#     sp = temp.drop_duplicates()[["SPY"]]
#     sp_changes = sp.pct_change().fillna(0)
#==============================================================================

    temp['Total_Value'] = temp.Total_Value.map(lambda x: np.round(x , 4))
    portvals = temp.drop_duplicates()[["Total_Value"]]
    

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
#==============================================================================
#     start_date = dt.datetime(2008,1,1)
#     end_date = dt.datetime(2008,6,1)
#     portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
#     portvals = portvals[['IBM']]  # remove SPY
#==============================================================================

    return portvals











