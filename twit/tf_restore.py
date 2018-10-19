from tensorflow import keras
import pandas as pd
import codecs
import my_lib


dataset = pd.DataFrame()

fprefix = ""

data_raw = pd.read_csv('testpos.csv', header=None, encoding='utf8', sep=';')[3]
dataset['Words'] = data_raw.apply(my_lib.splitstring)
dataset['Cat'] = 1


temp = pd.DataFrame()

data_raw = pd.read_csv('testneg.csv', header=None, encoding='utf8', sep=';')[3]
temp['Words'] = data_raw.apply(my_lib.splitstring)
temp['Cat'] = 0

dataset = dataset.append(temp, ignore_index=True)


test_data = dataset['Words']
test_labels = dataset['Cat']


with codecs.open('words with digits and e.txt', 'r', encoding = 'utf8')\
    as wordsfile: wds = wordsfile.readlines()

words = {}
index = 0
for i in wds:
    words.update({i[:-1]: index})
    index += 1

#print(words)

test_data = my_lib.WordReplace(test_data, words)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0,\
    padding='post', maxlen=256)  # , maxlen=256

#print(test_data)

# ML Magic

vocab_size = len(words)

print("Dict size: ", vocab_size)

model = keras.models.load_model('my_model_100epochs.h5')

results = model.evaluate(test_data, test_labels)
print(results)

#X = splitstring("Ого, на кинопоиске сделали саундтреки :D а я и не знала. Очень удобнено.")
#print(X)
#X = WordReplace([X], words)
#print(X)

#X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=256) # , maxlen=256
#print(X)

#inv_dict = {v: k for k, v in words.items()}
#print (WordReturn(X, inv_dict))
#print(words['<PAD>'])

#prediction = model.predict(X, batch_size=512, verbose=1)
#print(prediction)
