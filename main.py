import logging
import os
import random
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pymysql
import yaml
from faker import Faker
from gensim import models
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    handlers=[logging.FileHandler("logs" + os.sep + datetime.now().strftime('Recommendation_Engine_%d_%m_%Y.log')),
              logging.StreamHandler()],
    format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def data_import(type='fake', prod_info_path=None, rat_info_path=None, uhis_info_path=None, user=None, password=None,
                host=None, database=None, prod_SQL_Query=None, rat_SQL_Query=None, his_SQL_Query=None, arguments=None):
    if type == 'fake':
        faker = Faker()
        rating = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        data = []
        ################product_info
        sentence = faker.sentence()
        for i in range(100):
            temp = {}
            temp['prodId'] = i
            temp['name'] = faker.sentence(nb_words=3, variable_nb_words=True)[:-1]
            temp['description'] = faker.sentence()
            # temp['category'] = random.randrange(8)
            data.append(temp)
        prod_info = pd.DataFrame(data)
        pid = prod_info['prodId'].to_list()
        prod_info.to_csv('prod_info.csv')

        ################rating review
        data = []
        for i in range(2000):
            temp = {}
            temp['userId'] = random.randrange(100)
            temp['prodId'] = random.choice(pid)
            temp['product_review'] = faker.sentence()
            temp['rating'] = random.choice(rating)
            # temp['rating count']=random.randrange(1000)
            data.append(temp)
        rat_info = pd.DataFrame(data)
        rat_info.to_csv('rat_info.csv')

        #################users history
        data = []
        for i in range(1000):
            temp = {}
            # temp['timestamp'] = i
            temp['userId'] = random.randrange(100)
            temp['prodId'] = random.choice(pid)
            data.append(temp)
        uhis_info = pd.DataFrame(data)
        uhis_info.to_csv('uhis_info.csv')
        return prod_info, rat_info, uhis_info
    elif type == 'csv':
        prod_info = pd.read_csv(arguments['prod_info_path'],
                                usecols=[arguments['prodId'], arguments['name'], arguments['description']])
        prod_info.columns = ['prodId', 'name', 'description']
        rat_info = pd.read_csv(arguments['rat_info_path'],
                               usecols=[arguments['userId'], arguments['prodId'], arguments['rating'],
                                        arguments['product_review']])
        rat_info.columns = ['userId', 'prodId', 'rating', 'product_review']
        uhis_info = pd.read_csv(arguments['uhis_info_path'], usecols=[arguments['userId'], arguments['prodId']])
        uhis_info.columns = ['userId', 'prodId']
        return prod_info, rat_info, uhis_info
    elif type == 'mysql':
        try:
            dbcon = pymysql.connect(user, password, host, database)
            SQL_Query = pd.read_sql_query(prod_SQL_Query, dbcon)
            prod_info = pd.DataFrame(SQL_Query)
            SQL_Query = pd.read_sql_query(rat_SQL_Query, dbcon)
            rat_info = pd.DataFrame(SQL_Query)
            SQL_Query = pd.read_sql_query(his_SQL_Query, dbcon)
            uhis_info = pd.DataFrame(SQL_Query)
            return prod_info, rat_info, uhis_info
        except Exception as e:
            logging.error(f"Wunable to fetch the data from  {database}")
            exit(1)
    elif type == 'sqlite':
        try:
            sqlcon = sqlite3.connect(database)
            prod_info = pd.read_sql_query(prod_SQL_Query, sqlcon)
            rat_info = pd.read_sql_query(rat_SQL_Query, sqlcon)
            uhis_info = pd.read_sql_query(his_SQL_Query, sqlcon)
            return prod_info, rat_info, uhis_info
        except Exception as e:
            logging.error(f"unable to fetch the data from  {database}")
            exit(1)


def popular_products(uhis_info, prod_info):
    try:
        uhis_info['popularity'] = uhis_info.groupby('prodId')['prodId'].transform('count')
        # df=pd.read_csv('sample-data.csv', low_memory=False)

        uhis_info = uhis_info.drop_duplicates(subset='prodId', keep="last")
        prod_info['prodId'] = prod_info['prodId'].astype(str)
        uhis_info['prodId'] = uhis_info['prodId'].astype(str)
        prod_info = prod_info.merge(uhis_info[['prodId', 'popularity']], on=['prodId'], how='outer')

        prod_info['popularity'] = prod_info['popularity'].fillna(0)

        prod_info['popularity'] = (prod_info['popularity'] - prod_info[
            'popularity'].min()) / (
                                          prod_info['popularity'].max() - prod_info[
                                      'popularity'].min())
        uhis_info = uhis_info.drop(['popularity'], axis=1)
        return prod_info
    except Exception as e:
        logging.error(f"While counting popular_products {e}")
        exit(1)


def weighted_rating(x, m, C):
    """Formula:
        (v/(v+m) * R) + (m/(m+v) * C)

        ---


        v is the number of ratings
        m is the minimum rating count required to be listed in the data
        R is the average rating of the product
        C is the mean vote across the whole data.
        """
    v = x['rat_count']
    R = x['rat_avrg']
    return (v / (v + m) * R) + (m / (m + v) * C)


def weighted_average(prod_info, rat_info, only_qualified):
    try:
        rat_prod = rat_info.groupby(["prodId"], as_index=False).agg(
            rat_count=pd.NamedAgg(column="rating", aggfunc="count"),
            rat_avrg=pd.NamedAgg(column="rating", aggfunc="mean"))

        C = rat_prod['rat_avrg'].mean()

        m = rat_prod['rat_count'].quantile(0.90)

        """only qualified products"""
        if only_qualified:
            rat_prod = rat_prod.copy().loc[rat_prod['rat_count'] >= m]

        rat_prod['rscore'] = rat_prod.apply(weighted_rating, m=m, C=C, axis=1)
        rat_prod['rscore'] = (rat_prod['rscore'] - rat_prod['rscore'].min()) / (
                rat_prod['rscore'].max() - rat_prod['rscore'].min())
        rat_prod['prodId'] = rat_prod['prodId'].astype(str)
        prod_info_final = prod_info.merge(rat_prod, on='prodId', how='left')
        prod_info_final = prod_info_final.replace(np.nan, 0)
        return prod_info_final
    except Exception as e:
        logging.error(f"While counting weighted_average {e}")
        exit(1)


def preprocess(text):
    res_text = re.sub(r'[^\w\s]', '', text)
    return res_text


def sentiment_model_load(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # PT
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.save_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        logging.error(f"While loading model of sentiment {e}")
        exit(1)


def sentiment_apply(row, tokenizer, model):
    text = row['product_review']
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    row['negative'] = np.round(float(scores[0]), 4)
    row['neutral'] = np.round(float(scores[1]), 4)
    row['positive'] = np.round(float(scores[2]), 4)
    return row


def sentiment_review(rat_info, prod_info, model_path):
    try:
        tokenizer, model = sentiment_model_load(model_path)
        logging.info(f"sentiment analysis model loaded")
        rat_info = rat_info.apply(sentiment_apply, tokenizer=tokenizer, model=model, axis=1)

        rat_info = rat_info.groupby('prodId', as_index=False)[['negative', 'neutral', 'positive']].mean()
        rat_info['prodId'] = rat_info['prodId'].astype(str)
        prod_info = prod_info.merge(rat_info, on='prodId', how='left')
        prod_info = prod_info.replace(np.nan, 0)
        return prod_info
    except Exception as e:
        logging.error(f"While processing of sentiment analysis {e}")
        exit(1)


def vectorize(doc, w2v_model, stop_words=stop_words):
    doc = doc.lower()
    words = [w for w in doc.split(" ") if w not in stop_words]
    word_vecs = []
    for word in words:
        try:
            vec = w2v_model[word]
            word_vecs.append(vec)
        except KeyError:
            pass
    vector = np.mean(word_vecs, axis=0)
    return vector


def similar_products(prod_info, Embedding_file, stop_words=stop_words):
    try:
        prod_info.reset_index(inplace=True, drop=True)
        indexlist = prod_info['prodId'].to_dict()
        prod_info['textFeature_token'] = prod_info.apply(
            lambda row: word_tokenize(row['name'] + " " + row['description']), axis=1)
        prod_info['textFeature'] = prod_info.apply(lambda row: row['name'] + " " + row['description'], axis=1)

        w2v_model = models.KeyedVectors.load_word2vec_format(Embedding_file, binary=True)

        target_docs = prod_info['textFeature'].to_list()
        X = []
        for i in target_docs:
            X.append(vectorize(i, w2v_model, stop_words))

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)

        distances, indices = nbrs.kneighbors(X)

        distances_list = distances.tolist()
        indices_list = indices.tolist()

        indices_distances = {}
        for i in indexlist.keys():
            indices_distances[indexlist[i]] = list(zip([indexlist[j] for j in indices_list[i]], distances_list[i]))[1:]
        return pd.DataFrame(indices_distances.items(), columns=['prodId', 'Neighbors'])
    except Exception as e:
        logging.error(f"While counting of similar products {e}")
        exit(1)


def data_saved(type, prod_info, rat_info, uhis_info, indices_distances, host=None, port=None,
               user=None, password=None, database=None):
    try:
        if type == 'csv':
            prod_info.to_csv('prod_info_final.csv')
            # rat_info.to_csv('rat_info_final.csv')
            uhis_info.to_csv('uhis_info_final.csv')
            indices_distances.to_csv('indices_distances_final.csv')
        elif type == 'db':
            conn = pymysql.connect(host=host, port=port, user=user, password=password)
            conn.cursor().execute("CREATE DATABASE IF NOT EXISTS {0} ".format(database))
            # conn = pymysql.connect(host=host, port=port, password=password, db=database, charset='utf8')
            engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
            prod_info = prod_info.drop(['textFeature_token'], axis=1)
            prod_info = prod_info.drop(['textFeature'], axis=1)
            prod_info.to_sql(name='prod_info', con=engine, if_exists='replace', index=False)
            # rat_info.to_sql(name='rat_info', con=engine, if_exists='replace', index=False)
            uhis_info = uhis_info.drop(['popularity'], axis=1)
            uhis_info.to_sql(name='uhis_info', con=engine, if_exists='replace', index=False)
            indices_distances['Neighbors'] = indices_distances['Neighbors'].astype(str)
            indices_distances.to_sql(name='indices_distances', con=engine, if_exists='replace', index=False)
    except Exception as e:
        logging.error(f"While saving data {e}")
        exit(1)


if __name__ == '__main__':
    # read config yaml file
    f = open("rengine.config", 'r')
    arguments = yaml.load(f, Loader=yaml.FullLoader)
    logging.info(f"{arguments} configured from config file")

    # data import by 3 (type= [fake, csv, sql and sqlite])
    # for type='csv' provide prod_info_path, rat_info_path and uhis_info_path
    # for type='mysql' provide user, password, host, database for connection
    # and prod_SQL_Query, rat_SQL_Query, his_SQL_Query with proper column name
    # product_info: 'prodId', 'name', 'description', 'category'
    # rating_review_info: 'userId', 'prodId', 'product_review', 'rating'
    # user_history_info: 'timestamp', 'userId', 'prodId'
    logging.info(f"data fetching started from type {arguments['dtype']}")
    prod_SQL_Query = f"select {arguments['prodId']} as prodId, {arguments['name']} as name,{arguments['description']} as description from {arguments['prod_info']}"
    rat_SQL_Query = f"select {arguments['userId']} as userId, {arguments['prodId']} as prodId,{arguments['rating']} as rating,{arguments['product_review']} as product_review from {arguments['rat_info']} "
    his_SQL_Query = f"select {arguments['userId']} as userId, {arguments['prodId']} as prodId from {arguments['uhis_info']} "
    prod_info, rat_info, uhis_info = data_import(type=arguments['dtype'], database=arguments['database'],
                                                 prod_SQL_Query=prod_SQL_Query, rat_SQL_Query=rat_SQL_Query,
                                                 his_SQL_Query=his_SQL_Query, arguments=arguments)
    logging.info(f"data successfully fetched from type {arguments['dtype']}")

    #####################################
    ###        popular product        ###
    #####################################
    prod_info = popular_products(uhis_info, prod_info)
    logging.info(f"popular products counted")
    # print(prod_info.head())

    #####################################
    ###        weighted average       ###
    #####################################

    # only_qualified product will use based on the minimum rating count required to be listed in the data.
    prod_info = weighted_average(prod_info, rat_info, only_qualified=arguments['only_qualified'])
    logging.info(f"weighted average rating counted with {arguments['only_qualified']}")
    # print(prod_info.head())

    #####################################
    ###        Sentiment review       ###
    #####################################

    model_path = "twitter-roberta-base-sentiment"

    # tokenizer, model = sentiment_model_load(model_path)
    prod_info = sentiment_review(rat_info, prod_info, model_path)
    logging.info(f"sentiment analysis done on text review")

    #####################################
    ###        Similar products       ###
    #####################################

    # !wget -P /content/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

    Embedding_file = 'GoogleNews-vectors-negative300.bin.gz'
    indices_distances = similar_products(prod_info, Embedding_file, stop_words)
    logging.info(f"similar products counted for users")

    #####################################
    ###           Saved Data          ###
    #####################################
    # data_saved('csv', prod_info, rat_info, uhis_info, indices_distances, host='localhost', port=3306,
    #            user='hb', password='hbdev', database='demo')
    data_saved('db', prod_info, rat_info, uhis_info, indices_distances, host='localhost', port=3306,
               user='hb', password='hbdev', database=arguments['sdb'])
    logging.info(f"processed data saved successfully")
