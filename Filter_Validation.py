import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
stop = stopwords.words('english')
import numpy as np


data = pd.read_csv('Validation1.csv')
data = data[["target","comment_text","toxicity_annotator_count"]]
columns_titles = ["comment_text","toxicity_annotator_count","target"]
data=data.reindex(columns=columns_titles)

data['comment_text'] = data['comment_text'].apply(lambda x: x.lower())
data['comment_text'] = data['comment_text'].apply((lambda x: re.sub('[^a-zA-z\s]', '', x)))

print("word count before removing stop words")
data['word_count'] = data['comment_text'].apply(lambda x: len(x.split()))
#print(data['comment_text'])
data['comment_text'] = data['comment_text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

print("=========================================================================================")

print("word count after stop words")
#print(data['comment_text'])
data['word_count_stopwords'] = data['comment_text'].str.split().str.len()
print(data[['word_count','word_count_stopwords']])


voc_size = 80000
tokenizer = Tokenizer(num_words=voc_size, split=' ')
tokenizer.fit_on_texts(data['comment_text'].values)
X = tokenizer.texts_to_sequences(data['comment_text'].values)
X = pad_sequences(X)
print(X.shape[1])
import pickle
# https://androidkt.com/saved-keras-model-to-predict-text-from-scratch/
# save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


embed_dim= 256
batch_size = 64
lstm_out = 128


def createmodel():
    model = Sequential()
    model.add(Embedding(voc_size, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('lstm.h5')
    return model

sort_by_stop = data.sort_values('word_count_stopwords',ascending=False)
print(sort_by_stop)

X_final=np.array(X)
data['target'] = data['target'].fillna((data['target'].mean()))
y_final=np.array(data['target'])

print(X_final.shape, y_final.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

model = createmodel()
history= model.fit(X_train, Y_train, validation_split=0.33, epochs=5, batch_size=batch_size, verbose=2)
print(history.history)
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print(score)
print(acc)

print(history.history.keys())
#  "Accuracy"
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
