import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow.keras.backend as K
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention


def masked_loss_function(y_true, y_pred):
    mask_value = -1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    mse = keras.losses.MeanSquaredError()
    return mse(y_true * mask, y_pred * mask)


def build_CTS(pos_vocab_size, maxnum, maxlen, readability_feature_count,
              linguistic_feature_count, configs, output_dim):
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

    linguistic_input = layers.Input((linguistic_feature_count,), name='linguistic_input')
    readability_input = layers.Input((readability_feature_count,), name='readability_input')

    pos_hz_lstm_list = [layers.LSTM(lstm_units, return_sequences=True)(pos_avg_zcnn) for _ in range(output_dim)]
    pos_avg_hz_lstm_list = [Attention()(pos_hz_lstm) for pos_hz_lstm in pos_hz_lstm_list]
    pos_avg_hz_lstm_feat_list = [layers.Concatenate()([pos_rep, linguistic_input, readability_input])
                                 for pos_rep in pos_avg_hz_lstm_list]
    pos_avg_hz_lstm = tf.concat([layers.Reshape((1, lstm_units + linguistic_feature_count + readability_feature_count))(pos_rep)
                                 for pos_rep in pos_avg_hz_lstm_feat_list], axis=-2)

    final_preds = []
    for index, rep in enumerate(range(output_dim)):
        mask = np.array([True for _ in range(output_dim)])
        mask[index] = False
        non_target_rep = tf.boolean_mask(pos_avg_hz_lstm, mask, axis=-2)
        target_rep = pos_avg_hz_lstm[:, index:index+1]
        att_attention = layers.Attention()([target_rep, non_target_rep])
        attention_concat = tf.concat([target_rep, att_attention], axis=-1)
        attention_concat = layers.Flatten()(attention_concat)
        final_pred = layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    y = layers.Concatenate()([pred for pred in final_preds])

    model = keras.Model(inputs=[pos_word_input, linguistic_input, readability_input], outputs=y)

    model.summary()

    model.compile(loss=masked_loss_function, optimizer='rmsprop')

    return model
