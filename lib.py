# data libraries
import numpy as np
import pandas as pd
import yfinance as yf

# plotting libraries
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# to get frequencies of unevenly-spaced data
from astropy.timeseries import LombScargle

def get_data(tickers, period="max", interval="1d", filename="data.csv", threads=True):
    """ Get data from yahoo finance (TODO: store data persistently in csv or database) """
    
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        threads=threads
    )
    
    # drop incomplete rows and store in file
    data.dropna(inplace = True)
    data.to_csv(filename)

def line_charts(df, title, height=800, width=1280, subplots=True):
    """ Plot line charts of each column in a DataFrame.  Index is x variable """
    rows = df.shape[1]
    if subplots:
        fig = make_subplots(rows=rows, cols=1)
    else:
        fig = go.Figure()
    
    # Add traces
    for i in range(rows):
        if subplots:
            fig.append_trace(
                go.Scatter(x=df.index, y=df.iloc[:,i], name=df.columns[i]),
                row=i+1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(x=df.index, y=df.iloc[:,i], name=df.columns[i])
            )
        
    fig.update_layout(height=height, width=width, title_text=title)
    fig.show()

def calc_logreturns(df, drop=True):
    """
    Return a DataFrame containing log returns of the data in the passed frame.
    This works because the lambda function is applied to each column as a series.
    Shift makes it so each item in the series is divided by the previous one.
    """
    log_returns = df.apply(lambda x: np.log(x/x.shift(1)))
    return log_returns.dropna() if drop else log_returns

def calc_simplereturns(df, drop=True):
    """
    Return a DataFrame containing log returns of the data in the passed frame.
    This works because the lambda function is applied to each column as a series.
    Shift makes it so each item in the series is differenced with then divided by the previous one.
    """
    simple_returns = df.apply(lambda x: (x-x.shift(1))/x.shift(1))
    return simple_returns.dropna() if drop else simple_returns

def sliding_window(df, w):
    """
    Construct a sliding window point cloud of the input DataFrame.

    Calculates and returns an ndarray of point clouds taken from sliding windows of size w
    on the columns of the DataFrame.  Stride size is 1 and all possible windows are returned in
    chronological order.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing time series data
    w : int
        The window size for the sliding window to be applied to the data

    Returns
    -------
    numpy.ndarray
        An ndarray of shape (N - w + 1, w) representing N - w + 1 sliding windows of size w

    """
    data = df.to_numpy()
    # number of time steps considered
    N = data.shape[0]
    
    # use broadcasting to create a matrix of indices for N - w sliding windows of size w
    window_indices = np.arange(w).reshape(1,w) + np.arange(N - w + 1).reshape(N - w + 1,1)
    
    # Now index the ndarray and to create N - w point clouds and return the result
    return data[window_indices]

def low_pass_pgram(s, max_freq):
    """
    Return a tuple of ndarrays representing the Lomb-Scargle periodogram of the data stored in s
    for frequencies below max_freq
    """
    frequency, power = LombScargle(np.arange(s.size), s).autopower(method='cython')
    indicator = np.where(frequency<max_freq)
    
    return (frequency[indicator], power[indicator])