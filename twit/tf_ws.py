# import tensorflow as tf
from tensorflow import keras
# import pandas as pd
# import numpy as np
import codecs
from flask import Flask
from flask_restful import Resource, Api
from urllib.parse import unquote
import my_lib


with codecs.open('words with digits and e.txt','r', encoding = 'utf8')\
    as wordsfile: wds = wordsfile.readlines()

words = {}
index = 0
for i in wds:
    words.update({i[:-1]: index})
    index += 1

vocab_size = len(words)
print("Dict size: ", vocab_size)

inv_dict = {v: k for k, v in words.items()}

model = keras.models.load_model('my_model_100epochs.h5')


class _Predict(Resource):
    def get(self, X):
        return pred(X)


def pred(X):
    X = unquote(X)
    print ("Исходный запрос:")
    print (X)
    X = my_lib.splitstring(X)
    X = my_lib.WordReplace([X], words)
    print ("Восстановленный запрос:")
    my_lib.WordReturn(X, inv_dict)
    X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post',\
        maxlen=256) # , maxlen=256
    predict = float(model.predict(X, batch_size=512, verbose=0)[0][0])
    print("Предсказание: ", predict)

    if predict > 0.5:
        result = "Positive"
    else:
        result = "Negative"
    result = {"tonality": result}
    return result

# inv_dict = {v: k for k, v in words.items()}
# print (WordReturn(X, inv_dict))

pred("")

app = Flask(__name__)
api = Api(app)
api.add_resource(_Predict, '/predict/<X>')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5002')
