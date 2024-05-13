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
from tensorflow.keras.mixed_precision import set_global_policy

import time
import gc

from mixed_precision_utils import convert_to_mixed_precision, train_step

class TimeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        print(f"\nEpoch {epoch+1} time: {epoch_time:.2f} seconds")


print(tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # model = convert_to_mixed_precision(model, 'float16')

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = 8
batch_size = 100

start_data_loading = time.time()
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
end_data_loading = time.time()
data_loading_time = (end_data_loading - start_data_loading) * 1000
print(f"Data loading time: {data_loading_time:.2f} ms")
dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
dist_test_dataset = strategy.experimental_distribute_dataset(test_dataset)

steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_test) // batch_size

time_callback = TimeCallback()

start_time = time.time()

# for epoch in range(epochs):
#     epoch_loss_avg = tf.keras.metrics.Mean()
#     epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

#     for batch, (inputs, labels) in enumerate(dist_train_dataset):
#         loss, predictions = train_step(inputs, labels)
#         epoch_loss_avg.update_state(loss)
#         epoch_accuracy.update_state(labels, predictions)

#     val_loss_avg = tf.keras.metrics.Mean()
#     val_accuracy = tf.keras.metrics.BinaryAccuracy()

#     for batch, (inputs, labels) in enumerate(dist_test_dataset):
#         inputs = tf.cast(inputs, 'float16')
#         predictions = model(inputs, training=False)
#         loss = tf.keras.losses.binary_crossentropy(labels, predictions)
#         val_loss_avg.update_state(loss)
#         val_accuracy.update_state(labels, predictions)

#     print(f"Epoch {epoch+1}: Loss = {epoch_loss_avg.result():.4f}, Accuracy = {epoch_accuracy.result():.4f}, "
#           f"Val Loss = {val_loss_avg.result():.4f}, Val Accuracy = {val_accuracy.result():.4f}")
    
#     if epoch_loss_avg.result() < 0.0001:
#         break
    
history = model.fit(dist_train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                    validation_data=dist_test_dataset, validation_steps=validation_steps,
                    callbacks=[time_callback, EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
end_time = time.time()

training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

print(tf.config.list_physical_devices())

gc.collect()

