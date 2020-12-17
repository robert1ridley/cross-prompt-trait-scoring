import os
import time
import argparse
import random
import numpy as np
import tensorflow as tf
from configs.configs import Configs
from models.CTS_no_att import build_CTS_no_att
from utils.read_data import read_essays, read_pos_vocab
from utils.general_utils import get_scaled_down_scores, pad_hierarchical_text_sequences
from evaluators.multitask_evaluator_all_attributes import Evaluator as AllAttEvaluator


def main():
    parser = argparse.ArgumentParser(description="CTS no att model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--model_name', type=str,
                        choices=['multitask_split'],
                        help='name of model')
    args = parser.parse_args()
    test_prompt_id = args.test_prompt_id
    seed = args.seed

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Seed: {}".format(seed))

    configs = Configs()

    data_path = configs.DATA_PATH
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    features_path = configs.FEATURES_PATH
    readability_path = configs.READABILITY_PATH
    vocab_size = configs.VOCAB_SIZE
    epochs = configs.EPOCHS
    batch_size = configs.BATCH_SIZE

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
        'vocab_size': vocab_size
    }

    pos_vocab = read_pos_vocab(read_configs)
    train_data, dev_data, test_data = read_essays(read_configs, pos_vocab)

    max_sentlen = max(train_data['max_sentlen'], dev_data['max_sentlen'], test_data['max_sentlen'])
    max_sentnum = max(train_data['max_sentnum'], dev_data['max_sentnum'], test_data['max_sentnum'])
    print('max sent length: {}'.format(max_sentlen))
    print('max sent num: {}'.format(max_sentnum))
    train_data['y_scaled'] = get_scaled_down_scores(train_data['data_y'], train_data['prompt_ids'])
    dev_data['y_scaled'] = get_scaled_down_scores(dev_data['data_y'], dev_data['prompt_ids'])
    test_data['y_scaled'] = get_scaled_down_scores(test_data['data_y'], test_data['prompt_ids'])

    X_train_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
    X_dev_pos = pad_hierarchical_text_sequences(dev_data['pos_x'], max_sentnum, max_sentlen)
    X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)

    X_train_pos = X_train_pos.reshape((X_train_pos.shape[0], X_train_pos.shape[1] * X_train_pos.shape[2]))
    X_dev_pos = X_dev_pos.reshape((X_dev_pos.shape[0], X_dev_pos.shape[1] * X_dev_pos.shape[2]))
    X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))

    X_train_linguistic_features = np.array(train_data['features_x'])
    X_dev_linguistic_features = np.array(dev_data['features_x'])
    X_test_linguistic_features = np.array(test_data['features_x'])

    X_train_readability = np.array(train_data['readability_x'])
    X_dev_readability = np.array(dev_data['readability_x'])
    X_test_readability = np.array(test_data['readability_x'])

    Y_train = np.array(train_data['y_scaled'])
    Y_dev = np.array(dev_data['y_scaled'])
    Y_test = np.array(test_data['y_scaled'])

    print('================================')
    print('X_train_pos: ', X_train_pos.shape)
    print('X_train_readability: ', X_train_readability.shape)
    print('X_train_ling: ', X_train_linguistic_features.shape)
    print('Y_train: ', Y_train.shape)

    print('================================')
    print('X_dev_pos: ', X_dev_pos.shape)
    print('X_dev_readability: ', X_dev_readability.shape)
    print('X_dev_ling: ', X_dev_linguistic_features.shape)
    print('Y_dev: ', Y_dev.shape)

    print('================================')
    print('X_test_pos: ', X_test_pos.shape)
    print('X_test_readability: ', X_test_readability.shape)
    print('X_test_ling: ', X_test_linguistic_features.shape)
    print('Y_test: ', Y_test.shape)
    print('================================')

    model = build_CTS_no_att(len(pos_vocab), max_sentnum, max_sentlen, X_train_readability.shape[1],
                             X_train_linguistic_features.shape[1], configs, Y_train.shape[1])

    dev_features_list = [X_dev_pos, X_dev_linguistic_features, X_dev_readability]
    test_features_list = [X_test_pos, X_test_linguistic_features, X_test_readability]

    evaluator = AllAttEvaluator(test_prompt_id, dev_data['prompt_ids'], test_data['prompt_ids'], dev_features_list,
                                test_features_list, Y_dev, Y_test)

    evaluator.evaluate(model, -1, print_info=True)
    for ii in range(epochs):
        print('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time.time()
        model.fit(
            [X_train_pos, X_train_linguistic_features, X_train_readability],
            Y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time.time() - start_time
        print("Training one epoch in %.3f s" % tt_time)
        evaluator.evaluate(model, ii + 1)

    evaluator.print_final_info()


if __name__ == '__main__':
    main()
