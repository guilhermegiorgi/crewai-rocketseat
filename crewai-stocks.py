#IMPORT DAS LIBS
import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# In[137]:


#IMPORTANDO YAHOO FINANCE TOOL

def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2014-08-21", end="2024-08-21")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches crypto currency prices for {ticket} from the last 10 years about a specific crypto currency from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)


# In[138]:


# IMPORTANDO OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[139]:


stockPriceAnalyst = Agent(
    role= "Senior stock price Analyst",
    goal= "Find the {ticket} crypto currency price and trends analysis",
    backstory= """You're highly experienced in analyzing the price of an specific crypto currency
    and make predictions about its future price.""",
    verbose=True, 
    llm= llm,
    max_iter= 5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation = False, 
)


# In[140]:


getStockPrice = Task(
    description= "Analyse the crypto {ticket} price history and create a trend analysis of up, down and sideways" ,
    expected_output = """Specify the current trend crypto price - up, down and sideways. 
    eg. crypto= 'BTC-USD, price UP'
    """,
    agent= stockPriceAnalyst
)


# In[141]:


#IMPORTANDO A TOOL DE PESQUISA
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


# In[142]:


newsAnalyst = Agent(
    role= "Crypto News Analyst",
    goal= """Create a short summary of the market news related to the crypto {ticket}. Specify the current trend - up, down or sideways 
    with the news context. For each request crypto asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory= """You're highly experienced in analyzing the crypto market trends, blockchain news and have tracked assets for more than 15 years.
    
    You're also master level analyst in the crypto space and tradicional markets and have deep understanding of human psychology.
    
    You understand news, theirs titles and informations, but you look at those with a health dose of skepticism.
    You consider also the source of the news articles.
    """,
    verbose=True,
    llm= llm,
    max_iter= 10,
    memory= True, 
    tools=[search_tool],
    allow_delegation = False, 
)


# In[143]:


get_news = Task(
    description= f"""Take the crypto and always include BTC and GOLD to it(if not request).
    Use the search tool to search each one individually.
    
    The current date is {datetime.now()}.

    Compose the results into a helpfull report.""" ,
    expected_output = """A summary of the overeal market and one sentence summary for each request asset.
    Include a fear/greed index score for each assed based on the news. Use format: 
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED INDEX SCORE>
    """,
    agent= newsAnalyst
)


# In[144]:


stockAnalystWrite = Agent(
    role = "Crypto Analyst Writer",
    goal = """Write an insighfull compelling and informative 3 paragraph long newsletter based on the crypto report and price trend.""",
    backstory = """You're widely accepted as the best crypto analyst in the market. You understand complex concepts and create compelling stories and narratives 
    that resonate with audiences.
    
    You undestand macro factors and combine multiple theories - eg. cycle theory and fundamental analysis.
    You're able to hold multiple opinions when analyzing anything.""",
    verbose = True, 
    llm = llm,
    max_iter = 5,
    memory = True, 
    allow_delegation = True, 
)


# In[145]:


writeAnalysis = Task(
    description= """Use the crypto price trend and the stock news report to create an analysis and write the newsletter about the {ticket} that is brief
    and higlights the most important points.
    Focus on the crypto price trend, ns and fear/gree index score. That are the near future considerations?
    Include the previous analysis of crypto trend and nes summary.""",
    expected_output = """An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain: 
    
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the mear of the analysis includingg the news summary and fear/greed index score.
    - summary - key facts and concrete future trend prediction - up, down or sideways.  
    """,
    agent= stockAnalystWrite,
    context = [getStockPrice, get_news]
)

# # Tradução do relatório
# translateReport = Agent(
#     role = "Translator",
#     goal = "Translate the report into Portuguese.",
#     backstory = "You're an expert translator, fluent in both English and Portuguese.",
#     verbose = True, 
#     llm = llm,
#     max_iter = 5,
#     memory = False,
#     allow_delegation = False,
# )

# translateTask = Task(
#     description = "Translate the following report into Portuguese: {report}",
#     expected_output = "The translated report in Portuguese.",
#     agent = translateReport,
# )

crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, get_news, writeAnalysis],
    verbose = True,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)
with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])