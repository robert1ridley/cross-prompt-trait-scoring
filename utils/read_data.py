import pickle
import nltk
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils.general_utils import get_score_vector_positions

url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
MAX_SENTLEN = 50
MAX_SENTNUM = 100
pd.set_option('mode.chained_assignment', None)


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def shorten_sentence(sent, max_sentlen):
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            num = int(round(num))
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                num = int(np.ceil(num))
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
        return [tokens]

    return new_tokens


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        return tokens

    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        return sent_tokens
    else:
        raise NotImplementedError


def is_number(token):
    return bool(num_regex.match(token))


def read_word_vocab(read_configs):
    vocab_size = read_configs['vocab_size']
    file_path = read_configs['train_path']
    word_vocab_count = {}

    with open(file_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    for index, essay in enumerate(train_essays_list):
        content = essay['content_text']
        content = text_tokenizer(content, True, True, True)
        content = [w.lower() for w in content]
        for word in content:
            try:
                word_vocab_count[word] += 1
            except KeyError:
                word_vocab_count[word] = 1

    import operator
    sorted_word_freqs = sorted(word_vocab_count.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1

    word_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(word_vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        word_vocab[word] = index
        index += 1
    return word_vocab


def read_pos_vocab(read_configs):
    file_path = read_configs['train_path']
    pos_tags_count = {}

    with open(file_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    for index, essay in enumerate(train_essays_list[:16]):
        content = essay['content_text']
        content = text_tokenizer(content, True, True, True)
        content = [w.lower() for w in content]
        tags = nltk.pos_tag(content)
        for tag in tags:
            tag = tag[1]
            try:
                pos_tags_count[tag] += 1
            except KeyError:
                pos_tags_count[tag] = 1

    pos_tags = {'<pad>': 0, '<unk>': 1}
    pos_len = len(pos_tags)
    pos_index = pos_len
    for pos in pos_tags_count.keys():
        pos_tags[pos] = pos_index
        pos_index += 1
    return pos_tags


def get_readability_features(readability_path):
    with open(readability_path, 'rb') as fp:
        readability_features = pickle.load(fp)
    return readability_features


def get_linguistic_features(linguistic_features_path):
    features_df = pd.read_csv(linguistic_features_path)
    return features_df


def get_normalized_features(features_df):
    column_names_not_to_normalize = ['item_id', 'prompt_id', 'score']
    column_names_to_normalize = list(features_df.columns.values)
    for col in column_names_not_to_normalize:
        column_names_to_normalize.remove(col)
    final_columns = ['item_id'] + column_names_to_normalize
    normalized_features_df = None
    for prompt_ in range(1, 9):
        is_prompt_id = features_df['prompt_id'] == prompt_
        prompt_id_df = features_df[is_prompt_id]
        x = prompt_id_df[column_names_to_normalize].values
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_pd1 = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(normalized_pd1, columns=column_names_to_normalize, index = prompt_id_df.index)
        prompt_id_df[column_names_to_normalize] = df_temp
        final_df = prompt_id_df[final_columns]
        if normalized_features_df is not None:
            normalized_features_df = pd.concat([normalized_features_df,final_df],ignore_index=True)
        else:
            normalized_features_df = final_df
    return normalized_features_df


def read_essay_sets(essay_list, readability_features, normalized_features_df, pos_tags):
    out_data = {
        'essay_ids': [],
        'pos_x': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        sent_tag_indices = []
        tag_indices = []
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                tags = nltk.pos_tag(sent)
                for tag in tags:
                    if tag[1] in pos_tags:
                        tag_indices.append(pos_tags[tag[1]])
                    else:
                        tag_indices.append(pos_tags['<unk>'])
                sent_tag_indices.append(tag_indices)
                tag_indices = []

        out_data['pos_x'].append(sent_tag_indices)
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)
        if out_data['max_sentnum'] < len(sent_tag_indices):
            out_data['max_sentnum'] = len(sent_tag_indices)
    assert(len(out_data['pos_x']) == len(out_data['readability_x']))
    print(' pos_x size: {}'.format(len(out_data['pos_x'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


def read_essay_sets_word_flat(essay_list, readability_features, normalized_features_df, vocab):
    out_data = {
        'essay_ids': [],
        'words': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_essay_len': -1,
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        indices = []
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                for word in sent:
                    if is_number(word):
                        indices.append(vocab['<num>'])
                    elif word in vocab:
                        indices.append(vocab[word])
                    else:
                        indices.append(vocab['<unk>'])
        out_data['words'].append(indices)
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)
        if out_data['max_essay_len'] < len(indices):
            out_data['max_essay_len'] = len(indices)
    assert(len(out_data['words']) == len(out_data['readability_x']))
    print(' word_x size: {}'.format(len(out_data['words'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


def read_essay_sets_word(essay_list, readability_features, normalized_features_df, vocab):
    out_data = {
        'essay_ids': [],
        'words': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        sent_indices = []
        indices = []
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                for word in sent:
                    if is_number(word):
                        indices.append(vocab['<num>'])
                    elif word in vocab:
                        indices.append(vocab[word])
                    else:
                        indices.append(vocab['<unk>'])
                sent_indices.append(indices)
                indices = []
        out_data['words'].append(sent_indices)
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)
        if out_data['max_sentnum'] < len(sent_indices):
            out_data['max_sentnum'] = len(sent_indices)
    assert(len(out_data['words']) == len(out_data['readability_x']))
    print(' word_x size: {}'.format(len(out_data['words'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


def read_essay_sets_single_score(essay_list, readability_features, normalized_features_df, pos_tags, attribute_name):
    out_data = {
        'pos_x': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        if attribute_name in essay.keys():
            y = int(essay[attribute_name])
            out_data['data_y'].append([y])
            item_index = np.where(readability_features[:, :1] == essay_id)
            item_row_index = item_index[0][0]
            item_features = readability_features[item_row_index][1:]
            out_data['readability_x'].append(item_features)
            feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
            feats_list = feats_df.values.tolist()[0][1:]
            out_data['features_x'].append(feats_list)
            sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
            sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

            sent_tag_indices = []
            tag_indices = []
            for sent in sent_tokens:
                length = len(sent)
                if length > 0:
                    if out_data['max_sentlen'] < length:
                        out_data['max_sentlen'] = length
                    tags = nltk.pos_tag(sent)
                    for tag in tags:
                        if tag[1] in pos_tags:
                            tag_indices.append(pos_tags[tag[1]])
                        else:
                            tag_indices.append(pos_tags['<unk>'])
                    sent_tag_indices.append(tag_indices)
                    tag_indices = []

            out_data['pos_x'].append(sent_tag_indices)
            out_data['prompt_ids'].append(essay_set)
            if out_data['max_sentnum'] < len(sent_tag_indices):
                out_data['max_sentnum'] = len(sent_tag_indices)
    assert(len(out_data['pos_x']) == len(out_data['readability_x']))
    print(' pos_x size: {}'.format(len(out_data['pos_x'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


def read_essay_sets_single_score_words(essay_list, readability_features, normalized_features_df, vocab, attribute_name):
    out_data = {
        'words': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        if attribute_name in essay.keys():
            y = int(essay[attribute_name])
            out_data['data_y'].append([y])
            item_index = np.where(readability_features[:, :1] == essay_id)
            item_row_index = item_index[0][0]
            item_features = readability_features[item_row_index][1:]
            out_data['readability_x'].append(item_features)
            feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
            feats_list = feats_df.values.tolist()[0][1:]
            out_data['features_x'].append(feats_list)
            sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
            sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

            sent_indices = []
            indices = []
            for sent in sent_tokens:
                length = len(sent)
                if length > 0:
                    if out_data['max_sentlen'] < length:
                        out_data['max_sentlen'] = length
                    for word in sent:
                        if is_number(word):
                            indices.append(vocab['<num>'])
                        elif word in vocab:
                            indices.append(vocab[word])
                        else:
                            indices.append(vocab['<unk>'])
                    sent_indices.append(indices)
                    indices = []

            out_data['words'].append(sent_indices)
            out_data['prompt_ids'].append(essay_set)
            if out_data['max_sentnum'] < len(sent_indices):
                out_data['max_sentnum'] = len(sent_indices)
    assert(len(out_data['words']) == len(out_data['readability_x']))
    print(' words size: {}'.format(len(out_data['words'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


def read_essays_words_flat(read_configs, word_vocab):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_word_flat(
        train_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    dev_data = read_essay_sets_word_flat(
        dev_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    test_data = read_essay_sets_word_flat(
        test_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    return train_data, dev_data, test_data


def read_essays_words(read_configs, word_vocab):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_word(
        train_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    dev_data = read_essay_sets_word(
        dev_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    test_data = read_essay_sets_word(
        test_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    return train_data, dev_data, test_data


def read_essays(read_configs, pos_tags):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets(train_essays_list, readability_features, normalized_linguistic_features, pos_tags)
    dev_data = read_essay_sets(dev_essays_list, readability_features, normalized_linguistic_features, pos_tags)
    test_data = read_essay_sets(test_essays_list, readability_features, normalized_linguistic_features, pos_tags)
    return train_data, dev_data, test_data


def read_essays_single_score(read_configs, pos_tags, attribute_name):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_single_score(
        train_essays_list, readability_features, normalized_linguistic_features, pos_tags, attribute_name)
    dev_data = read_essay_sets_single_score(
        dev_essays_list, readability_features, normalized_linguistic_features, pos_tags, attribute_name)
    test_data = read_essay_sets_single_score(
        test_essays_list, readability_features, normalized_linguistic_features, pos_tags, attribute_name)
    return train_data, dev_data, test_data


def read_essays_single_score_words(read_configs, word_vocab, attribute_name):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_single_score_words(
        train_essays_list, readability_features, normalized_linguistic_features, word_vocab, attribute_name)
    dev_data = read_essay_sets_single_score_words(
        dev_essays_list, readability_features, normalized_linguistic_features, word_vocab, attribute_name)
    test_data = read_essay_sets_single_score_words(
        test_essays_list, readability_features, normalized_linguistic_features, word_vocab, attribute_name)
    return train_data, dev_data, test_data
