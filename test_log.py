from bson import ObjectId
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['miraway']
col = db['log']
cur = col.find()
for c in cur:
    print(c['result'], c['time'])