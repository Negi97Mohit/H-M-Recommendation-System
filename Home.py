import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from IPython.display import Image, HTML
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
# import kaggle
import os ,glob, warnings
warnings.filterwarnings('ignore')
from pandas_profiling import ProfileReport

import plotly.figure_factory as ff
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# pd.options.display.max_colwidth=1200
# pd.options.display.max_columns= None
# pd.options.display.max_rows=50

from streamlit_pandas_profiling import st_profile_report

#Loading dataset
transaction_data=pd.read_csv(r"transactions_train.csv")
articles=pd.read_csv(r"articles.csv")
customers=pd.read_csv(r"customers.csv")

transaction_data.head(5)

articles.head(5)

customers.head(5)

transaction_profile = ProfileReport(transaction_data, title="Transaction Profiling Report")
st_profile_report(transaction_profile)

fig_prod_type = px.histogram(articles,x='product_type_name')
# fig_prod_type.show()
# Plot!
st.plotly_chart(fig_prod_type, use_container_width=True)

fig_prod_group = px.histogram(articles,y='product_group_name',color='product_type_name')

# fig_prod_group.show()
st.plotly_chart(fig_prod_group, use_container_width=True)

fig_dept_name = px.histogram(articles,y = 'department_name')
fig_dept_name.show()
articles.shape[0]-articles['article_id'].nunique()
fig_section_name = px.histogram(articles, y='section_name')
# fig_section_name.show()
st.plotly_chart(fig_section_name, use_container_width=True)


fig_colors=px.histogram(articles,y='colour_group_name',color='perceived_colour_value_name')
st.plotly_chart(fig_colors, use_container_width=True)
# fig_colors.show()


fig_customers_age= px.histogram(customers,x='age')
# fig_customers_age.show()
st.plotly_chart(fig_customers_age, use_container_width=True)


customers['Active'].isnull().value_counts(), customers.shape[0]#,customers['Active'].count() 
fig_cust_club_status = px.histogram(customers,y='club_member_status')
st.plotly_chart(fig_cust_club_status, use_container_width=True)

# fig_cust_club_status.show()
fig_fashion_news_freq = px.histogram(customers,y='fashion_news_frequency')

st.plotly_chart(fig_fashion_news_freq, use_container_width=True)
# fig_fashion_news_freq.show()


article=pd.read_csv(r"articles.csv")
cols=article.columns.tolist()
article.prod_name=article.prod_name.astype('str')
article.detail_desc=article.detail_desc.astype('str')
st.write(article.head(5))

name_corpus=' '.join(article.prod_name)
desc_corpus=' '.join(article.detail_desc)
article['content']=article[['prod_name','detail_desc']].astype('str').apply(lambda x: '//'.join(x),axis=1)
random_article=article.sample(frac=0.1, replace=True, random_state=1)
random_article.content.fillna('Null',inplace=True)

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(random_article.content)
tfidf_matrix.shape
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cols.append('content')
random_article=pd.DataFrame(np.array(random_article),columns=cols)

result={}
for idx,row in random_article.iterrows():
    
    sim_indices=cosine_sim[idx].argsort()[:-100:-1]
    sim_items=[(cosine_sim[idx][i],random_article['article_id'][i]) for i in sim_indices]
    result[row['article_id']]=sim_items[1:]


def item(article_id):
    name=random_article.loc[random_article['article_id']==article_id]['content'].tolist()[0].split('//')[0]
    code=random_article.loc[random_article['article_id']==article_id]['product_code']
    # desc   = '\nDescription:'+ random_article.loc[random_article['article_id'] == article_id]['content'].tolist()[0].split(' // ')[1][0:200] + '...'
    prediction=name+str(code)
    return prediction

def recomment(article_id,num):
    
    print("Recommentding:"+str(num)+'similar product to '+item(article_id))
    print('----')
    recs=result[article_id][:num]
    for rec in recs:
        print('\nRecommended: ' + item(rec[1]) + '\n(score:' + str(rec[0]) + ')')
        
st.tile("Reccomendations are :")
st.write(recomment(753708001,25))

article_profile = ProfileReport(articles, title="Transaction Profiling Report")
st_profile_report(article_profile)

#Dataset for first rating
val=pd.DataFrame()
val=(transaction_data.groupby(['article_id']).count()).t_dat
val=pd.DataFrame(val)
val.rename(columns={'t_dat':'count'},inplace=True)
val.reset_index(inplace=True)
val.head(5)

