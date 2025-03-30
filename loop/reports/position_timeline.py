from turtle import back


def position_timeline(backtester_object, test_id):

    '''Creates a line graph where each change is position in respect 
    to the USDT valua where that position change took place.
    
    backtest_object | object | BackTester object
    test_id | int | the id for the test to be plotted

    '''

    import astetik

    data = backtester_object.account_dfs[test_id]
    profit = backtester_object.profit_df.loc[test_id]['profit']
    profit_rate = backtester_object.profit_df.loc[test_id]['profit_rate']

    astetik.line(data, 
                 x=['buy_price_usdt', 'sell_price_usdt', 'open'],
                 markerstyle=['o', 'x', '|'],
                 markersize=8,
                 title='Profit :\$' + str(profit) + ' (' + str(profit_rate) + '\%)',
                 sub_title='',
                 linestyle=['-', '-', '--'],
                 linewidth=[0, 0, 2],
                 palette=["green", "red", "grey"],
                 legend=True,
                 legend_labels=['BUY', 'SELL', 'MARKET'],
                 y_limit=[data[1:]['open'].min() - 10, data[1:]['open'].max() + 10],
                 x_label='Minutes',
                 y_label='USDT')
