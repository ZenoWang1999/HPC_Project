import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding

def convert_to_mixed_precision(model, dtype):
    for layer in model.layers:
        if isinstance(layer, (Dense, LSTM, Embedding)):
            layer.kernel = layer.kernel.astype(dtype)
            if layer.use_bias:
                layer.bias = layer.bias.astype(dtype)
    return model

@tf.function
def train_step(model, inputs, labels, optimizer):
    inputs = tf.cast(inputs, 'float16')
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions