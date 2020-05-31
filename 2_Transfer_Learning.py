from keras.models import load_model
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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
import pickle

model = load_model('lstm.h5')
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)



data = pd.read_csv('Validation2.csv')
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

#max_fatures = 4000
loaded_tokenizer.fit_on_texts(data['comment_text'].values)
X = loaded_tokenizer.texts_to_sequences(data['comment_text'].values)
X = pad_sequences(X)
print(X.shape[1])

embed_dim= 256
batch_size = 64
lstm_out = 128


sort_by_stop = data.sort_values('word_count_stopwords',ascending=False)
print(sort_by_stop)


X_final=np.array(X)
data['target'] = data['target'].fillna((data['target'].mean()))
y_final=np.array(data['target'])
print(X_final.shape, y_final.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
model.fit(X_train, Y_train, epochs=5, batch_size=batch_size, verbose=2)
model.save('lstm1.h5')
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print(score)
print(acc)

