import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

import time
import gc

df=pd.read_csv('preprocessed_data.csv')
label = LabelEncoder()
df['sentiment'] = label.fit_transform(df['sentiment'])

max_words = 500
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['preprocessed_review'])
X = tokenizer.texts_to_sequences(df['preprocessed_review'])
X = pad_sequences(X, maxlen=max_len)
y = np.array(df['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=420)


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 8
batch_size = 100

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_test) // batch_size

start_time = time.time()

history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                    validation_data=test_dataset, validation_steps=validation_steps,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")
gc.collect()
