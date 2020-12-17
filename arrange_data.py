import os
import pickle


def combine_all_prompt_essays(file_list, essays, prompt):
    if prompt < 3:
        attribute_score_indices = {
            'score': 3,
            'content': 4,
            'organization': 5,
            'word_choice': 6,
            'sentence_fluency': 7,
            'conventions': 8
        }
    else:
        attribute_score_indices = {
            'score': 3,
            'content': 4,
            'prompt_adherence': 5,
            'language': 6,
            'narrativity': 7
        }
    for file_ in file_list:
        input_file = open(file_, 'r')
        lines = input_file.readlines()
        for line in lines[1:]:
            tokens = line.strip().split('\t')
            essay = {
                'essay_id': tokens[0],
                'prompt_id': tokens[1]
            }
            for key in attribute_score_indices.keys():
                essay[key] = tokens[attribute_score_indices[key]]
            essays.append(essay)
    return essays


def combine_for_prompt_seven_eight(filepath, essays, prompt):
    if prompt == 7:
        attribute_score_indices = {
            'content': (10, 16),
            'organization': (11, 17),
            'style': (12, 18),
            'conventions': (13, 19)
        }
    elif prompt == 8:
        attribute_score_indices = {
            'content': (10, 16, 22),
            'organization': (11, 17, 23),
            'voice': (12, 18, 24),
            'word_choice': (13, 19, 25),
            'sentence_fluency': (14, 20, 26),
            'conventions': (15, 21, 27)
        }

    with open(filepath, 'r', encoding='latin-1') as input_file:
        for index, line in enumerate(input_file):
            tokens = line.strip().split('\t')
            if index == 0:
                pass
            else:
                essay = {
                    'essay_id': tokens[0],
                    'prompt_id': tokens[1],
                    'score': tokens[6]
                }
                if prompt == 7:
                    if int(tokens[1]) == prompt:
                        for key in attribute_score_indices.keys():
                            rater1 = int(tokens[attribute_score_indices[key][0]])
                            rater2 = int(tokens[attribute_score_indices[key][1]])
                            resolved_score = rater1 + rater2
                            essay[key] = resolved_score
                        essays.append(essay)
                elif prompt == 8:
                    if int(tokens[1]) == prompt:
                        for key in attribute_score_indices.keys():
                            rater1 = int(tokens[attribute_score_indices[key][0]])
                            rater2 = int(tokens[attribute_score_indices[key][1]])
                            attribute_tokens = tokens[10:28]
                            if len(attribute_tokens) == 12:
                                resolved_score = rater1 + rater2
                            elif len(attribute_tokens) == 18:
                                resolved_score = int(tokens[attribute_score_indices[key][2]]) * 2
                            else:
                                raise NotImplementedError
                            essay[key] = resolved_score
                        essays.append(essay)
    return essays


def find_matches(list_of_essay_dicts, tsv_path):
    matched_essays = []
    input_file = open(tsv_path, 'r')
    lines = input_file.readlines()
    for line in lines[1:]:
        tokens = line.strip().split('\t')
        essay_id = tokens[0]
        prompt_id = int(tokens[1])
        content = tokens[2]
        matched = False
        for essay_dict in list_of_essay_dicts:
            if essay_dict['essay_id'] == essay_id:
                matched = True
                essay_dict['content_text'] = content
                matched_essays.append(essay_dict)
                break
    return matched_essays


def main():
    attribute_essays_path_root = 'data/aes-cross-val-data-attributes-tsv/'
    essay_fold_data_path_root = 'data/PAES-data/'
    all_essays_path = 'data/training_set_rel3.tsv'
    write_path = 'data/cross_prompt_attributes/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    all_combined_attribute_essays = []
    for i in range(1, 7):
        prompt_attribute_essays_path = attribute_essays_path_root + str(i) + '/'
        prompt_attribute_train_path = prompt_attribute_essays_path + 'train.tsv'
        prompt_attribute_dev_path = prompt_attribute_essays_path + 'dev.tsv'
        prompt_attribute_test_path = prompt_attribute_essays_path + 'test.tsv'
        prompt_attribute_paths = [
            prompt_attribute_train_path, prompt_attribute_dev_path, prompt_attribute_test_path]

        all_combined_attribute_essays = combine_all_prompt_essays(
            prompt_attribute_paths, all_combined_attribute_essays, i)

    for i in range(7, 9):
        all_combined_attribute_essays = combine_for_prompt_seven_eight(
            all_essays_path, all_combined_attribute_essays, i)

    for i in range(1, 9):
        if not os.path.exists(write_path + str(i)):
            os.makedirs(write_path + str(i))
        print(f'matching prompt: {i}')
        fold_path = essay_fold_data_path_root + str(i) + '/'
        train_path = fold_path + 'train.tsv'
        dev_path = fold_path + 'dev.tsv'
        test_path = fold_path + 'test.tsv'

        cross_prompt_train = find_matches(all_combined_attribute_essays, train_path)
        cross_prompt_dev = find_matches(all_combined_attribute_essays, dev_path)
        cross_prompt_test = find_matches(all_combined_attribute_essays, test_path)

        with open(write_path + str(i) + '/train.pk', 'wb') as out_train_file:
            pickle.dump(cross_prompt_train, out_train_file)

        with open(write_path + str(i) + '/dev.pk', 'wb') as out_dev_file:
            pickle.dump(cross_prompt_dev, out_dev_file)

        with open(write_path + str(i) + '/test.pk', 'wb') as out_test_file:
            pickle.dump(cross_prompt_test, out_test_file)


if __name__ == '__main__':
    main()
