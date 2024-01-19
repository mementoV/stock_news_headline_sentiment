import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import re
import random
from newspaper import Article
from newspaper import Config
import yfinance as yf
import datetime
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
import plotly
import os 
import sys
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from timeit import default_timer as timer 
def treemap():
    sp500 = pd.read_html(r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    tickers = []
    deltas = []
    sectors =[]
    market_caps = []

    for ticker in sp500:

        try:
            ## create Ticker object
            stock = yf.Ticker(ticker)
            tickers.append(ticker)

            ## download info
            info = stock.info

            ## download sector
            sectors.append(info['sector'])

            ## download daily stock prices for 2 days
            hist = stock.history('2d')

            ## calculate change in stock price (from a trading day ago)
            deltas.append((hist['Close'][1]-hist['Close'][0])/hist['Close'][0])

            ## calculate market cap
            market_caps.append(info['sharesOutstanding'] * info['previousClose'])
        except Exception as e:
            tickers.pop()

    df = pd.DataFrame({'ticker':tickers,
                  'sector': sectors,
                  'delta': deltas,
                  'market_cap': market_caps,
                  })

    color_bin = [-1,-0.02,-0.01,0, 0.01, 0.02,1]
    df['colors'] = pd.cut(df['delta'], bins=color_bin, labels=['red','indianred','lightpink','lightgreen','lime','green'])
 
    return df

def upload_to_google_sheet():
    #url = 'https://docs.google.com/spreadsheets/d/12345'+id+'/export?format=csv'

    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    credentials = ServiceAccountCredentials.from_json_keyfile_name('stocks-treemap-91bcc2f94bcc.json', scope)
    client = gspread.authorize(credentials)

    spreadsheet = client.open('stocks news treemap')

    try:

        with open('treemap.csv', 'r') as file_obj:
            content = file_obj.read()
            client.import_csv(spreadsheet.id, data=content)
            print('Data uploaded to Google Sheets')
    except Exception as e:
        print(e)
if __name__ == '__main__':

    
    start_time = timer()
    df = treemap()
    df.to_csv('treemap.csv')
    upload_to_google_sheet()
    end_time = timer()
    print(f"Time: {end_time-start_time:.3f} seconds")
    print(f"Time: {(end_time-start_time)/60:.3f} min")
