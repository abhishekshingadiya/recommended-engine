import ast
import logging
import os
from datetime import datetime
from functools import wraps

import jwt
import pandas as pd
import pymysql
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    handlers=[logging.FileHandler("logs" + os.sep + datetime.now().strftime('Recommendation_APis_%d_%m_%Y.log')),
              logging.StreamHandler()],
    format='%(levelname)s: %(message)s', datefmt=' %I:%M:%S %p',
    level=logging.INFO)
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")
Rkey = os.getenv("Rkey")
app = Flask("API_with_mysql")
logging.info(f"APIs integrated with host={host}, user={user} and db={db}")
JWT_SECRET_KEY = 'hbdev'


###################  TOKEN VERIFICATION  ###################

def token_required(to):
    @wraps(to)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        # return 401 if token is not passed
        if not token:
            return jsonify({'message': 'Token is missing !!'}), 401

        try:
            # decoding the payload to fetch the stored details
            data = jwt.decode(token, JWT_SECRET_KEY, algorithms="HS256")

            if 'Rkey' in data:
                if data['Rkey'] == Rkey:
                    return to(data)
                else:
                    return jsonify({
                        'message': f'Rkey is not valid'
                    }), 401
            else:
                return False
        except Exception as e:
            return jsonify({
                'message': f'Token is invalid or expired !! {e}'
            }), 401

    return decorated


###################  Trending products  ###################

@app.route('/tred_item', methods=['GET', 'POST'])
@token_required
def tred_item(to):
    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
        )
        cur = conn.cursor()
        cur.execute(f"select prodId from prod_info order by popularity DESC limit 10;")
        output = cur.fetchall()
        t_products = []
        for prod in output:
            t_products.append(prod[0])
        return jsonify({'prodIds': t_products})
    except Exception as e:
        print(f"While counting trending products {e}")


###################  HYBRID RECOMMENDED PRODUCT  ###################

@app.route('/pred_item', methods=['GET', 'POST'])
@token_required
def hybrid_recommandation(to):
    try:
        uid = to["uid"]
        pids = to["pids"]
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
        )
        cur = conn.cursor()
        recommanded = []
        cur.execute(f"select Neighbors from indices_distances where prodId in {tuple(pids)}")
        output = cur.fetchall()
        for i in output:
            recommanded.extend(ast.literal_eval(i[0]))
        r_products = {}
        for prod in recommanded:
            if prod[0] in r_products:
                r_products[prod[0]] = min(r_products[prod[0]], prod[1])
            else:
                r_products[prod[0]] = prod[1]
        cur.execute(f"select prodId,count(*) as count from uhis_info where userId={uid} group by prodId;")
        output = cur.fetchall()
        oldvisited = {}
        for i in output:
            oldvisited[i[0]] = i[1]
        fr_products = {}
        for key, value in r_products.items():
            if key not in pids:
                if key in oldvisited.keys():
                    fr_products[key] = abs(value - max(r_products.values())) * oldvisited[key]
                else:
                    fr_products[key] = abs(value - max(r_products.values()))
        fr_productsdf = pd.DataFrame(list(fr_products.items()), columns=['prodId', 'similarity'])

        fr_productsdf['similarity'] = (fr_productsdf['similarity'] - fr_productsdf['similarity'].min()) / (
                fr_productsdf['similarity'].max() - fr_productsdf['similarity'].min())
        fr_productsdf['similarity'] = fr_productsdf['similarity'].apply(lambda x: x * 0.2)
        temp = pd.read_sql(
            f"select `prodId`, `popularity`, `rscore`, `positive` from prod_info where prodId in {tuple(fr_productsdf['prodId'])}",
            con=conn)
        # To close the connection
        conn.close()
        fr_productsdf['prodId'] = fr_productsdf['prodId'].astype(str)
        temp['prodId'] = temp['prodId'].astype(str)
        fr_productsdf = fr_productsdf.merge(temp, on='prodId')[
            ['prodId', 'similarity', 'popularity', 'rscore', 'positive']]
        fr_productsdf['popularity'] = fr_productsdf['popularity'].apply(lambda x: x * 0.1)
        fr_productsdf['rscore'] = fr_productsdf['rscore'].apply(lambda x: x * 0.3)
        fr_productsdf['positive'] = fr_productsdf['positive'].apply(lambda x: x * 0.4)
        fr_productsdf['Final_score'] = fr_productsdf.iloc[:, -4:-1].sum(axis=1)
        fr_productsdf['Final_score'] = fr_productsdf['Final_score'].apply(lambda x: x * 100)

        recommanded_prodcuts = fr_productsdf.sort_values('Final_score', ascending=False)[['prodId', 'Final_score']]
        prod_dict = []

        for index, row in list(recommanded_prodcuts.iterrows()):
            prod_dict.append(dict(row))

        return jsonify(prod_dict)
    except Exception as e:
        print(f"While counting recommended products {e}")


###################  HIGHLY RATED PRODUCT  ###################

@app.route('/hrat_item', methods=['GET', 'POST'])
@token_required
def hrat_item(to):
    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
        )
        cur = conn.cursor()
        cur.execute(f"select prodId from prod_info order by rscore DESC limit 10;")
        output = cur.fetchall()
        h_products = []
        for prod in output:
            h_products.append(prod[0])
        return jsonify({'prodIds': h_products})
    except Exception as e:
        print(f"While counting high rated products {e}")


###################  ALSO VIEWED THIS PRODUCTS  ###################
@app.route('/avt_item', methods=['GET', 'POST'])
@token_required
def avt_item(to):
    try:
        pid = to["pid"]
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
        )
        cur = conn.cursor()
        cur.execute(
            f"SELECT c.prodId, COUNT(*) FROM `uhis_info` a JOIN `uhis_info` b ON a.prodId=b.prodId JOIN `uhis_info` c ON b.userId=c.userId WHERE a.`prodId`={pid} AND c.prodId!={pid} GROUP BY c.prodId ORDER BY 2 DESC limit 10;")
        output = cur.fetchall()
        avt_products = []
        for prod in output:
            avt_products.append(prod[0])
        return jsonify({'prodIds': avt_products})
    except Exception as e:
        print(f"While finding also viewed this products {e}")


###################  SIMILAR PRODUCTS  ###################

@app.route('/sim_item', methods=['GET', 'POST'])
@token_required
def sim_item(to):
    try:
        pid = to["pid"]
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
        )
        cur = conn.cursor()
        cur.execute(f"select Neighbors from indices_distances where prodId ={pid}")
        output = cur.fetchall()
        s_products = []
        for prod in ast.literal_eval(output[0][0]):
            s_products.append(prod[0])
        return jsonify({'prodIds': s_products})
    except Exception as e:
        print(f"While finding similar products {e}")


if __name__ == '__main__':
    app.run(host="192.168.37.19", port=5000)
