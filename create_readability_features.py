from utils.read_data import text_tokenizer
import readability
import pickle
import numpy as np


def main():
    features_data_file = 'data/allreadability.pickle'
    features_object = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: []
    }
    unwanted_features = [
        'paragraphs',
        'words',
        'characters',
        'sentences_per_paragraph',
        'words_per_sentence',
    ]
    final_array = None
    data_file_path = 'data/training_set_rel3.tsv'
    data = open(data_file_path, encoding="ISO-8859-1")
    lines = data.readlines()
    data.close()
    for index, line in enumerate(lines[1:]):
        if index % 50 == 0:
            print(f"processed {index} essays")
        tokens = line.strip().split('\t')
        essay_id = int(tokens[0])
        essay_set = int(tokens[1])
        content = tokens[2].strip()
        score = tokens[6]
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sentences = [' '.join(sent) + '\n' for sent in sent_tokens]
        sentences = ''.join(sentences)
        readability_scores = readability.getmeasures(sentences, lang='en')
        features = [essay_id]
        for cat in readability_scores.keys():
            for subcat in readability_scores[cat].keys():
                if subcat not in unwanted_features:
                    ind_score = readability_scores[cat][subcat]
                    features.append(ind_score)
        features_object[essay_set].append(features)
    for key in features_object.keys():
        features_object[key] = np.array(features_object[key])
        min_v, max_v = features_object[key].min(axis=0), features_object[key].max(axis=0)
        features = (features_object[key] - min_v) / (max_v - min_v)
        features = np.nan_to_num(features)
        features = features_object[key]
        features_object[key][:, 1:] = features[:, 1:]
        if isinstance(final_array, type(None)):
            final_array = features_object[key]
        else:
            final_array = np.vstack((final_array, features_object[key]))

    with open(features_data_file, 'wb') as fp:
        pickle.dump(final_array, fp)


if __name__ == "__main__":
    main()