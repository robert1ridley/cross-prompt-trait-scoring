import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow.keras.backend as K
from custom_layers.zeromasking import ZeroMaskedEntries


def masked_loss_function(y_true, y_pred):
    mask_value = -1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    mse = keras.losses.MeanSquaredError()
    return mse(y_true * mask, y_pred * mask)


def build_AES_aug_multitask(vocab_size, maxlen, configs, embedding_weights, output_dim):
    embedding_dim = configs.EMBEDDING_DIM
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS
    dropout_prob = configs.DROPOUT

    word_input = layers.Input(shape=(maxlen,), dtype='int32', name='word_input')
    x = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxlen,
                         weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = layers.Dropout(dropout_prob, name='drop_x')(x_maskedout)
    zcnn = layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid', name='zcnn')(drop_x)
    hz_lstm = layers.LSTM(lstm_units, return_sequences=True, name='hz_lstm')(zcnn)
    avg_hz_lstm = layers.GlobalAveragePooling1D(name='avg_hz_lstm')(hz_lstm)

    y = layers.Dense(units=output_dim, activation='sigmoid', name='y_att')(avg_hz_lstm)

    model = keras.Model(inputs=word_input, outputs=y)

    model.summary()

    model.compile(loss=masked_loss_function, optimizer='rmsprop')

    return model
