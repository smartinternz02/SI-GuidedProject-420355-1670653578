import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import flask
from flask import Flask,render_template, request
import pickle


app = Flask(__name__) # initializing a flask app
model=pickle.load(open("restaurant.pkl",'rb'))  #loading the model

#loading the updated dataset
zomato_df=pd.read_csv("restaurant1.csv")


@app.route('/')# route to display the home page
def home():
    return render_template('home.html')#rendering the home page


@app.route('/extractor')
def extractor():
    return render_template('extractor.html')

#extractor page
@app.route('/keywords',  methods=['POST'])# route to show the predictions in a web UI
def keywords():
    #typ=request.form['type']
    output=request.form['output']
    #if typ=="text":
        #output=re.sub("[^a-zA-Z.,]"," ",output)
    print(output)
    print(type(output))
    df_percent = zomato_df.sample(frac=0.5)
    df_percent.set_index('name', inplace=True)
    indices = pd.Series(df_percent.index)
    # Creating tf-idf matrix
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'].fillna(' '))
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
     
    def recommend(name, cosine_similarities = cosine_similarities):
        
        # Create a list to put top restaurants
        recommend_restaurant = []
    
        # Find the index of the hotel entered
        idx = indices[indices == name].index[0]
        
        # Find the restaurants with a similar cosine-sim value and order them from bigges number
        score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
        
        # Extract top 30 restaurant indexes with a similar cosine-sim value
        top30_indexes = list(score_series.iloc[0:31].index)
        
        # Names of the top 30 restaurants
        for each in top30_indexes:
            recommend_restaurant.append(list(df_percent.index)[each])
            
        # Creating the new data set to show similar restaurants
        df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
      
        # Create the top 30 similar restaurants with some of their columns
        for each in recommend_restaurant:
            df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
        
        # Drop the same named restaurants and sort only the top 10 by the highest rating
        df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
        pd.set_option('display.max_columns', None)

        df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
        print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
        
        return df_new
    
    result = recommend(output)
    print(result)
    print(type(result))
    #print(result[0])
        
    #print(result[0])
   # res = result.to_string(index=False)

    

    #showing the prediction results in a UI
    return render_template('keywords.html',keyword=result.to_html())

     
if __name__ == "__main__":
   # running the app
    app.run(debug=False)
