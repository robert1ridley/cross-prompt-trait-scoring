import tensorflow.keras.layers as layers
from tensorflow import keras
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention


def build_PAES(pos_vocab_size, maxnum, maxlen, readability_feature_count, linguistic_feature_count,
               configs):
    pos_embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS

    pos_word_input = layers.Input(shape=(maxnum*maxlen,), dtype='int32', name='pos_word_input')
    pos_x = layers.Embedding(output_dim=pos_embedding_dim, input_dim=pos_vocab_size, input_length=maxnum*maxlen,
                            weights=None, mask_zero=True, name='pos_x')(pos_word_input)
    pos_x_maskedout = ZeroMaskedEntries(name='pos_x_maskedout')(pos_x)
    pos_drop_x = layers.Dropout(dropout_prob, name='pos_drop_x')(pos_x_maskedout)
    pos_resh_W = layers.Reshape((maxnum, maxlen, pos_embedding_dim), name='pos_resh_W')(pos_drop_x)
    pos_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='pos_zcnn')(pos_resh_W)
    pos_avg_zcnn = layers.TimeDistributed(Attention(), name='pos_avg_zcnn')(pos_zcnn)
    pos_hz_lstm = layers.LSTM(lstm_units, return_sequences=True, name='pos_hz_lstm')(pos_avg_zcnn)
    pos_avg_hz_lstm = Attention(name='pos_avg_hz_lstm')(pos_hz_lstm)

    # Add linguistic features
    linguistic_input = layers.Input((linguistic_feature_count,), name='linguistic_input')

    # Add Readability features
    readability_input = layers.Input((readability_feature_count,), name='readability_input')

    final_output = layers.Concatenate()([pos_avg_hz_lstm, linguistic_input, readability_input])

    y = layers.Dense(units=1, activation='sigmoid', name='y_att')(final_output)

    model = keras.Model(inputs=[pos_word_input, linguistic_input, readability_input], outputs=y)

    model.summary()

    model.compile(loss='mse', optimizer='rmsprop')

    return model
