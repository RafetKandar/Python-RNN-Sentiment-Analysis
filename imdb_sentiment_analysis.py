# -*- coding: utf-8 -*-
"""
eng : import library
tr : kütüphaneleri yüklüyoruz
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers.embeddings import Embedding 
from keras.layers import SimpleRNN, Dense, Activation

"""
eng : load data and examining
tr : veriyi yüklüyoruz ve inceliyoruz.
"""
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "ibdb.npz",
               num_words = None,
               skip_top = 0, 
               maxlen = None, 
               seed = 113, 
               start_char = 1,
               index_from = 3)

print("Type: ", type(X_train))

print("X train shape : ", X_train.shape)
print("Y train shape : ", Y_train.shape)

print("Y train values: ", np.unique(Y_train))
print("Y test values: ", np.unique(Y_test))

unique, counts = np.unique(Y_train, return_counts = True)
print("Y train distrubution : ", dict(zip(unique, counts)))


unique, counts = np.unique(Y_test, return_counts = True)
print("Y test distrubution : ", dict(zip(unique, counts)))

plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freg")
plt.title("Y train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freg")
plt.title("Y test")


d = X_train[0]
print(d)
print(len(d))

"""
eng : we look at the length of the sentences.
tr : cümlelerin uzunluğuna bakıyoruz.
"""
review_len_train = []
review_len_test = []
for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))


sns.distplot(review_len_train, hist_kws = {"alpha":0.3})
sns.distplot(review_len_test, hist_kws = {"alpha":0.3})

print("Train Mean : ",np.mean(review_len_train))
print("Train Median : ",np.median(review_len_train))
print("Train Mode : ",stats.mode(review_len_train))

"""
eng : number of word
tr : sayıların kelime karşılıkları
"""
word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

"""
eng : word equivalent of the number
tr : sayısal değerin hangi kelimeye karşılık geldiğine bakıyoruz.
"""
for keys, values in word_index.items():
    if values == 22:
        print(keys)

"""
eng : We are looking at the full review
tr : sayısal değerleri kelime karşılıklarına çevirerek yorumun tamamına bakıyoruz.
"""

def whatIsSay(index = 24):
    
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])
    print(decode_review)
    print(Y_train[index])
    return decode_review

decoded_review = whatIsSay(36)


"""
eng : preprocess
tr : veriyi model için uygun hale getiriyoruz.
"""
num_words = 15000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words = num_words)

maxlen = 130
X_train = pad_sequences(X_train, maxlen = maxlen)
X_test = pad_sequences(X_test, maxlen = maxlen)

print(X_train[5])

for i in X_train[0:10]:
    print(len(i))

decoded_review = whatIsSay(5)

"""
eng : Create RNN
tr : RNN modelini oluşturuyoruz.
"""
rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length = len(X_train[0])))
rnn.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = False, activation = "relu"))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])


"""
eng : fit model
tr : modelimizi eğitiyoruz.
"""
history = rnn.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 15, batch_size = 128, verbose = 1)

score = rnn.evaluate(X_test, Y_test)
print("Accuracy : %",score[1]*100)

"""
eng : Visualize loss and acc
tr : loss ve acc sonuçlarımızı görselleştiriyoruz.
"""
plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Accuracy")
plt.ylabel("Acc")
plt.xlabel("epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Loss")
plt.ylabel("Acc")
plt.xlabel("epochs")
plt.legend()
plt.show()

