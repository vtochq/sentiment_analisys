import tensorflow as tf
from tensorflow import keras
import pandas as pd
import codecs
import my_lib
import datetime
import pickle


infprefix = ""
savefprefix = "test_act_"
regenerate_data = True

if regenerate_data:
    print("Loading Positive ", datetime.datetime.now())
    dataset = pd.DataFrame()

    data_raw = pd.read_csv(infprefix+'positive.csv', header=None, encoding='utf8',\
        sep=';')[3]
    dataset['Words'] = data_raw.apply(my_lib.splitstring)
    dataset['Cat'] = 1


    print("Loading Negative ", datetime.datetime.now())
    temp = pd.DataFrame()

    data_raw = pd.read_csv(infprefix+'negative.csv', header=None, encoding='utf8',\
        sep=';')[3]
    temp['Words'] = data_raw.apply(my_lib.splitstring)
    temp['Cat'] = 0

    dataset = dataset.append(temp, ignore_index=True)

    train_data = dataset['Words']
    train_labels = dataset['Cat']

    # print(dataset)

    print("Creating dictionary ", datetime.datetime.now())
    words = my_lib.WordsDic(train_data)

    print("Replace words to digits ", datetime.datetime.now())
    train_data = my_lib.WordReplace(train_data, words)

    print ("Padding matrix ", datetime.datetime.now())
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0,\
        padding='post', maxlen=256)  # , maxlen=256

    # print(len(train_data))

    # save dictionary
    with codecs.open(savefprefix+'words.txt', 'w', encoding = 'utf8')\
        as wordfile: wordfile.writelines(i + '\n' for i in words)
    # save train_data
    with open(savefprefix+'objs.pkl', 'wb') as f:
        pickle.dump([train_data, train_labels], f)
else:
    # loading dictionary
    with codecs.open(savefprefix+'words.txt', 'r', encoding = 'utf8')\
        as wordsfile: wds = wordsfile.readlines()
    words = {}
    index = 0
    for i in wds:
        words.update({i[:-1]: index})
        index += 1

    # loading train_data
    with open(savefprefix+'objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        train_data, train_labels = pickle.load(f)

# ML Magic

vocab_size = len(words)

print("Dict size: ", vocab_size)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 8))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(8, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

val_size = round(len(train_data)*0.1)  # 10% of train_data

# x_val = train_data[:3].reset_index(drop=True)
x_val = train_data[:val_size]
partial_x_train = train_data[val_size:]

y_val = train_labels[:val_size]
partial_y_train = train_labels[val_size:]
'''
print("Start learning", datetime.datetime.now())
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=1,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# model.save_weights('model.save')
model.save(savefprefix+'model.h5')
'''
print(type(partial_x_train))
print(partial_x_train)
