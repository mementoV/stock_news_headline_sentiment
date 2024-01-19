import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
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
from plotly import graph_objects as go
from streamlit_gsheets import GSheetsConnection
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import time
import finnhub

def moving_average(df,prices,n):
  return df[prices].rolling(n).mean()

def treemap(df,day1,day2):
    """
    This function is used to plot a treemap of the sp500.
    :param df: dataframe
    :return: None
    """
    # Get the current date
    current_date = datetime.now()

    # Calculate yesterday's date
    yesterday_date = current_date - timedelta(days=1)
    yesterday_date = yesterday_date.strftime('%Y-%m-%d')
    d_tmp = current_date - timedelta(days=2)
    d_tmp = d_tmp.strftime('%Y-%m-%d')
    
    st.subheader(f"Change % from {day2} to {day1}")
    #plot the treemap
    fig = px.treemap(df, path=[ 'sector','ticker'], values = 'market_cap', color='colors',
                    color_discrete_map ={'(?)':'#262931', 'red':'red', 'indianred':'indianred','lightpink':'lightpink', 'lightgreen':'lightgreen','lime':'lime','green':'green'},

                    custom_data=['delta']
                    )
    fig.update_traces(textinfo="label+text", textfont_size=12,texttemplate="<b>%{label}</b> <br>%{customdata[0]:.2p}")
    fig.update_traces(textposition="middle center")
    fig.update_layout(margin = dict(t=30, l=10, r=10, b=10), font_size=20)

    st.plotly_chart(fig)

def plt_data(df):
    """
    This function is used to plot the data of a time series.
    :param df: dataframe
    :return: None
    """
    ma100 = moving_average(df,'Close',100)
    ma200 = moving_average(df,'Close',200)
    #plot the evolution of the stock price (Close)
    st.subheader("Evolution of the close price for the last 30 days")
    fig,ax = plt.subplots(nrows=1,ncols=1)
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Date[-30:], df.Close[-30:],color='green',label='Close price')
    plt.plot(df.Date[-30:], ma100[-30:],color='blue',label='100 moving average')
    plt.plot(df.Date[-30:], ma200[-30:],color='red',label='200 moving average')
    plt.legend()
    st.pyplot(fig)

    #plot the evolution of the stock price
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close',line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.Date, y=ma100, name='100 ma',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.Date, y=ma200, name='200 ma',line=dict(color='red')))
    fig.layout.update(title_text='Stock Price',xaxis_rangeslider_visible=True,yaxis=dict(
       autorange = True,
       fixedrange= False,
   ))
    fig.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig)

def get_page(url:str):
  """
  This function is used to get the html code from the url
  :param url: str
  :return: BeautifulSoup object
  """
  hdr={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36','Accept': 'text/html'}
  response = requests.get(url,headers=hdr)

  if response.status_code ==200:
    time.sleep(10)
    page = BeautifulSoup(response.content,'html.parser')
    
    return page

  else:
    st.error("Error in getting the page")
    return None
  
def finviz_news(ticker: str):
    """
    This function is used to get the info of headlines from finviz
    :param ticker: str
    :return: dataframe
    """
    news = list()
    date = list()
    url = list()
    finviz_news_page = get_page('https://finviz.com/quote.ashx?t=' + ticker)
    finviz_table = finviz_news_page.find_all("table",class_="fullview-news-outer news-table")

    tr_tag = finviz_table[0].findAll("tr")

    # Using try because somehow getting Nonetype error for "table.a" during cloud version
    for table in tr_tag:
       try:
            tmp_a = table.a.text
            tmp_t = table.td.text
            if type(tmp_a) is not None and type(tmp_t) is not None:
                    
                news.append(table.a.text)
                date.append(table.td.text.strip())
                url.append(table.a.get('href'))

       except Exception as e:
           print(e)


    df =  pd.DataFrame({'date':date,'Headline':news,'url':url})
    return df

def yahoo_news(ticker: str):
    """
    This function is used to get the info of headlines from yahoo finance
    :param ticker: str
    :return: dataframe
    """
    title = list()
    publisher = list()
    url = list()
    info = yf.Ticker(ticker)
    news = info.get_news()

    for i in range(len(news)):
        title.append(news[i]['title'])
        publisher.append(news[i]['publisher'])
        url.append(news[i]['link'])

    return pd.DataFrame({'publisher':publisher,'Headline':title,'url':url})


def read_gsheet(csv_name: str):
   """
   This function is used to read the data from the gsheet
   :return: dataframe
   """
   connect = st.connection('gsheets',type=GSheetsConnection)
   df = connect.read(worksheet=csv_name,usecols=[1,2,3,4,5],nrows =501)
   return df

def finnnhub_info(ticker: str):
   try:
        
        key = st.secrets["finn_api_key"]
        finnhub_client = finnhub.Client(api_key=key)
        info = finnhub_client.company_profile2(symbol=ticker)
        info.pop('logo')
        info.pop('finnhubIndustry')
        st.subheader(f'Information about "{ticker}"')
        st.write(info)

   except Exception as e:
        print(e)

def news_analysis(ticker,finviz_news,yahoo_news):
    """
    This function is used to analyze the sentiments of the news of the stock
    :param ticker: str
    :param finviz_news: dataframe
    :param yahoo_news: dataframe
    """
    
    
    #Import pre-trained model from huggingface transformers for sentiment analysis
    #and tokenize the text for the inputs


    
    
    sentiments_score = dict()

    for i in range(20):
       finviz_text = finviz_news.iloc[i]['Headline']
       inputs = tokenizer(finviz_text, return_tensors="pt", padding=True)
       outputs = model(**inputs)[0]
       val = labels[np.argmax(outputs.detach().numpy())]

       sentiments_score.setdefault(val,0)
       sentiments_score[val] += 1

    try:
        for i in range(len(yahoo_news)):
            yahoo_text = yahoo_news.iloc[i]['Headline']
            inputs = tokenizer(yahoo_text, return_tensors="pt", padding=True)
            outputs = model(**inputs)[0]
            val = labels[np.argmax(outputs.detach().numpy())]

            sentiments_score.setdefault(val,0)
            sentiments_score[val] += 1

    except Exception as e:
        print(e)
    h = "Hugging face 🤗"
    link = "https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    st.write(f'"{ticker}" Sentiment Analysis with the recents headline news above with')
    st.markdown(f"<a href={link}>{h}</a>",unsafe_allow_html=True)
    st.write(sentiments_score)

    




if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    labels = {1:'neutral', 2:'positive',0:'negative'}
    with st.sidebar:
       st.subheader('A user-friendly website designed to showcase stock information from S&P 500 index and conduct stock price analysis, incorporating sentiment analysis of headline news.')
       st.write('Made by Sakda Heng')
    sp500 = pd.read_html(r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

    csv_name = "stocks news treemap"

    ############################
    connect = st.connection('gsheets',type=GSheetsConnection)
    treemap_df = connect.read(worksheet=csv_name,usecols=[1,2,3,4,5],nrows =501)
    #############################
    #treemap_df = read_gsheet(csv_name)    
    start_date = '2012-01-01'
    current_date = datetime.now()
    
    sp500.tolist()
    st.title('Stock Price with headline news sentiment analysis')

    ticker = st.selectbox('Select ticker', sp500)
    #ticker = st.text_input('Enter a stock ticker','SPY')
    df = yf.download(ticker,start_date)
    df = df.reset_index()
    day1 = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    day2 = df['Date'].iloc[-2].strftime('%Y-%m-%d')


    treemap(treemap_df,day1,day2)
    finnnhub_info(ticker)

    
    st.subheader(f'"{ticker}" Prices')
    st.write(df.tail())

    st.subheader(f'"{ticker}" summary')
    st.write(df[['Open','Close','Low','High','Volume']].describe())

    plt_data(df)
    
    #Display some news and sentiments analysis
    st.subheader(f'"{ticker}" Finviz News')
    finviz_news = finviz_news(ticker)
    st.write(finviz_news)

    try:
        yahoo_news = yahoo_news(ticker)
        st.subheader(f'"{ticker}" Yahoo News')
        st.write(yahoo_news)
    except Exception as e:
        print(e)

    news_analysis(ticker,finviz_news,yahoo_news)

    prompt = st.text_area('Enter a sentence then use Ctrl+Enter to analyze')

    input = tokenizer(prompt, return_tensors="pt", padding=True)
    output = model(**input)[0]
    val = labels[np.argmax(output.detach().numpy())]
    if len(prompt) > 0:
        st.write(f'The sentiment is {val.upper()}')





