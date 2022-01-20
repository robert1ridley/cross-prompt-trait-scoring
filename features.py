import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import spacy
import csv
import nltk
from nltk.corpus import brown, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import time
import textstat
from utils.read_data import text_tokenizer

nltk.download('brown')
set_words = set(brown.words())
nlp = spacy.load('en_core_web_sm')


class FeatureSet:
    def __init__(self, text, id, prompt_number, score):
        self.id = id
        self.prompt_number = prompt_number
        self.score = score
        self.raw_text = text
        self.raw_sentences = nltk.sent_tokenize(text)
        self.sentences = []
        self.words = []
        for sentence in self.raw_sentences:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            self.sentences.append(sentence)
            sent_words = nltk.word_tokenize(sentence)
            self.words.extend(sent_words)
        self.p = []
        self.p2 = []
        self.word_count = len(self.words)
        self.char_count = 0
        self.mean_word_length = 0
        self.word_length_variance = 0
        self.mean_sentence_length = 0
        self.sentence_length_variance = 0
        self.comma_and_prep = 0
        self.unique_words = 0
        self.spacy_clause_number = 0
        self.spacy_max_clauses_in_sentence = 0
        self.spacy_mean_clause_length = 0
        self.spacy_mean_clauses_per_sent = 0
        self.spelling_mistake_count = 0
        self.average_sentence_depth = 0
        self.average_leaf_depth = 0
        self.spacy_average_sentence_depth = 0
        self.spacy_average_leaf_depth = 0

        # readability
        self.syllable_count = 0
        self.flesch_reading_ease = 0
        self.flesch_kincaid_grade = 0
        self.fog_scale = 0
        self.smog = 0
        self.automated_readability = 0
        self.coleman_liau = 0
        self.linsear_write = 0
        self.dale_chall_readability = 0
        self.text_standard = 0

        # additional features
        self.stop_prop = 0
        self.punc_pos_proportions = {}
        self.positive_sentence_prop = 0
        self.negative_sentence_prop = 0
        self.neutral_sentence_prop = 0
        self.overall_positivity_score = 0
        self.overall_negativity_score = 0


    def get_readability_features(self):
        sent_tokens = text_tokenizer(self.raw_text, replace_url_flag=True, tokenize_sent_flag=True)
        sentences = [' '.join(sent) + '\n' for sent in sent_tokens]
        sentences = ''.join(sentences)
        self.syllable_count = textstat.syllable_count(sentences)
        self.flesch_reading_ease = textstat.flesch_reading_ease(sentences)
        self.flesch_kincaid_grade = textstat.flesch_kincaid_grade(sentences)
        self.fog_scale = textstat.gunning_fog(sentences)
        self.smog = textstat.smog_index(sentences)
        self.automated_readability = textstat.automated_readability_index(sentences)
        self.coleman_liau = textstat.coleman_liau_index(sentences)
        self.linsear_write = textstat.linsear_write_formula(sentences)
        self.dale_chall_readability = textstat.dale_chall_readability_score(sentences)
        self.text_standard = textstat.text_standard(sentences)


    def get_stopword_proportion(self):
        total_words = self.word_count
        removed = [word for word in self.words if word.lower() not in stopwords.words('english')]
        filtered_count = len(removed)
        self.stop_prop = filtered_count/total_words


    def get_word_sentiment_proportions(self):
        sentiment_intensity_analyzer = SentimentIntensityAnalyzer()
        sentence_count = len(self.sentences)
        positive_sentences = 0
        negative_sentences = 0
        neutral_sentences = 0
        accumulative_sentiment = 0
        for sentence in self.sentences:
            ss = sentiment_intensity_analyzer.polarity_scores(sentence)
            if ss['compound'] > 0:
                positive_sentences += 1
            elif ss['compound'] < 0:
                negative_sentences += 1
            else:
                neutral_sentences += 1
            accumulative_sentiment += ss['compound']
        average_accumulative_sentiment = accumulative_sentiment / sentence_count

        self.positive_sentence_prop = positive_sentences / sentence_count
        self.negative_sentence_prop = negative_sentences / sentence_count
        self.neutral_sentence_prop = neutral_sentences / sentence_count
        if average_accumulative_sentiment > 0:
            self.overall_positivity_score = 1 - average_accumulative_sentiment
        elif average_accumulative_sentiment < 0:
            self.overall_negativity_score = 0 - average_accumulative_sentiment


    def spacy_parse(self):
        sentences = self.raw_sentences
        for sentence in sentences:
            self.p2.append(nlp(sentence))


    def calculate_mean_word_length(self):
        for word in self.words:
            self.char_count += len(word)
        self.mean_word_length = self.char_count/self.word_count


    def calculate_word_length_variance(self):
        squared_diff_sum = 0
        for word in self.words:
            diff = len(word) - self.mean_word_length
            squared_diff = diff * diff
            squared_diff_sum += squared_diff
        self.word_length_variance = squared_diff_sum / self.word_count


    def calculate_mean_sentence_length(self):
        self.mean_sentence_length = len(self.words) / len(self.sentences)


    def calculate_sentence_length_variance(self):
        squared_diff_sum = 0
        for sentence in self.sentences:
            sent_length = len(nltk.word_tokenize(sentence))
            diff = sent_length - self.mean_sentence_length
            squared_diff = diff * diff
            squared_diff_sum += squared_diff
        self.sentence_length_variance = squared_diff_sum / len(self.sentences)


    def count_punctuation_and_pos(self):
        punc_and_pos_count = \
        {
            ',': 0,
            '.': 0,
            'VB': 0,
            'JJR': 0,
            'WP': 0,
            'PRP$': 0,
            'VBN': 0,
            'VBG': 0,
            'IN': 0,
            'CC': 0,
            'JJS': 0,
            'PRP': 0,
            'MD': 0,
            'WRB': 0,
            'RB': 0,
            'VBD': 0,
            'RBR': 0,
            'VBZ': 0,
            'NNP': 0,
            'POS': 0,
            'WDT': 0,
            'DT': 0,
            'CD': 0,
            'NN': 0,
            'TO': 0,
            'JJ': 0,
            'VBP': 0,
            'RP': 0,
            'NNS': 0
        }
        tag_count = 0
        sentences = self.raw_sentences
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            tags = nltk.pos_tag(words)
            for tag in tags:
                tag_count += 1
                if tag[1] in punc_and_pos_count.keys():
                    punc_and_pos_count[tag[1]] += 1
        self.comma_and_prep = punc_and_pos_count['IN'] + punc_and_pos_count[',']
        for key in punc_and_pos_count:
            self.punc_pos_proportions[key] = punc_and_pos_count[key] / tag_count


    def unique_word_count(self):
        word_counts = {}
        self.unique_words = 0
        for word in self.words:
            if word not in word_counts.keys():
                word_counts[word] = 1
            else:
                word_counts[word] += 1
        for w in word_counts:
            if word_counts[w] == 1:
                self.unique_words += 1


    def spacy_clause_count(self):
        clause_word_count = 0
        for parsed_sentence in self.p2:
            sentence_clause_count = 0
            for token in parsed_sentence:
                if token.dep_ == 'relcl':
                    self.spacy_clause_number += 1
                    sentence_clause_count += 1
                    this_clause = list(w.text_with_ws for w in token.subtree)
                    clause_word_count += len(this_clause)
            if sentence_clause_count > self.spacy_max_clauses_in_sentence:
                self.spacy_max_clauses_in_sentence = sentence_clause_count
        try:
            self.spacy_mean_clause_length = clause_word_count / self.spacy_clause_number
        except ZeroDivisionError:
            self.spacy_mean_clause_length = 0
        try:
            self.spacy_mean_clauses_per_sent = self.spacy_clause_number / len(self.sentences)
        except ZeroDivisionError:
            self.spacy_mean_clauses_per_sent = 0


    def spelling_mistakes(self):
        punctuation = set(string.punctuation)
        text = ''.join([w for w in self.raw_text.lower() if w not in punctuation])
        tokens = nltk.word_tokenize(text)
        self.spelling_mistake_count = len([word for word in tokens if word not in set_words and '@' not in word])


    def spacy_parser_depth(self):
        parser_depth_count = 0
        leaf_count = 0
        leaf_depth_count = 0
        for parsed_sentence in self.p2:
            root = []
            word_and_head = {}
            sentence_deepest_node = -1
            for token in parsed_sentence:
                word_and_head[token.idx] = token.head.idx
                if token.idx == token.head.idx:
                    root.append(token.idx)
            for word in word_and_head:
                leaf_count += 1
                current_word = word
                count = 0
                while current_word not in root:
                    count += 1
                    current_word = word_and_head[current_word]
                if count > sentence_deepest_node:
                    sentence_deepest_node = count
                leaf_depth_count += count
            parser_depth_count += sentence_deepest_node
            self.spacy_average_sentence_depth = parser_depth_count / len(self.sentences)
            self.spacy_average_leaf_depth = leaf_depth_count / leaf_count


def write_to_csv(csv_file_path, data):
    with open(csv_file_path, 'w') as outfile:
        print("Writing to csv")
        fp = csv.DictWriter(outfile, data[0].keys())
        fp.writeheader()
        fp.writerows(data)


if __name__ == '__main__':
    all_essays = []
    with open('data/training_set_rel3.tsv', encoding="latin-1") as input_file:
        next(input_file)
        for index, line in enumerate(input_file):
            if index % 50 == 0:
                print(f"Processed: {index} essays")
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[6])

            feature_set = FeatureSet(content, essay_id, essay_set, score)
            feature_set.get_readability_features()
            feature_set.calculate_mean_word_length()
            feature_set.calculate_word_length_variance()
            feature_set.calculate_mean_sentence_length()
            feature_set.calculate_sentence_length_variance()
            feature_set.count_punctuation_and_pos()
            feature_set.unique_word_count()
            feature_set.spacy_parse()
            feature_set.spacy_clause_count()
            feature_set.spelling_mistakes()
            feature_set.spacy_parser_depth()
            feature_set.get_stopword_proportion()
            feature_set.get_word_sentiment_proportions()
            feature_set_dict = {
                'item_id': feature_set.id,
                'prompt_id': feature_set.prompt_number,
                'mean_word': feature_set.mean_word_length,
                'word_var': feature_set.word_length_variance,
                'mean_sent': feature_set.mean_sentence_length,
                'sent_var': feature_set.sentence_length_variance,
                'ess_char_len': feature_set.char_count,
                'word_count': feature_set.word_count,
                'prep_comma': feature_set.comma_and_prep,
                'unique_word': feature_set.unique_words,
                'clause_per_s': feature_set.spacy_mean_clauses_per_sent,
                'mean_clause_l': feature_set.spacy_mean_clause_length,
                'max_clause_in_s': feature_set.spacy_max_clauses_in_sentence,
                'spelling_err': feature_set.spelling_mistake_count,
                'sent_ave_depth': feature_set.spacy_average_sentence_depth,
                'ave_leaf_depth': feature_set.spacy_average_leaf_depth,
                'automated_readability': feature_set.automated_readability,
                'linsear_write': feature_set.linsear_write,
                'stop_prop': feature_set.stop_prop,
                'positive_sentence_prop': feature_set.positive_sentence_prop,
                'negative_sentence_prop': feature_set.negative_sentence_prop,
                'neutral_sentence_prop': feature_set.neutral_sentence_prop,
                'overall_positivity_score': feature_set.overall_positivity_score,
                'overall_negativity_score': feature_set.overall_negativity_score
            }
            feature_set_dict.update(feature_set.punc_pos_proportions)
            feature_set_dict['score'] = feature_set.score
            all_essays.append(feature_set_dict)

            # TODO:
            # unique bigrams count
            # unique trigrams count


    write_to_csv('data/hand_crafted_v3.csv', all_essays)