# Cross-prompt Trait Scoring

This repository contains the code used to produce the results from the paper _Automated_ _Cross-prompt_ _Scoring_ _of_ 
_Essay_ _Traits_ published in _AAAI_ _2021_.

## System Requirements
This code uses _python_ _3.6_ along with the following packages and version numbers:
- tensorflow=2.0.0
- numpy=1.18.1
- nltk=3.4.5
- pandas=1.0.5
- scikit-learn=0.22.1

Ensure that these packages have been added to the virtual environment before running the project code.

## Running the Code
Each model is trained through the use of a bash script. Run the models as follows (for _Hi_ _att_ and _AES_ _aug_, 
GloVe embeddings are required. Download `glove.6B.50d.txt` from https://nlp.stanford.edu/projects/glove/ and place 
in the `embeddings` directory):
- To run the _Hi_ _att_ model, run `./train_Hi_att.sh`
- To run the _AES_ _aug_ model, run `./train_AES_aug.sh`
- To run the _PAES_ model, run `./train_PAES.sh`
- To run the _CTS_ _no_ _att_ model, run `./train_CTS_no_att.sh`
- To run the _CTS_ model, run `./train_CTS.sh`

The bash scripts will run each model on each of the prompts and traits a total of 5 times, 
once for each of the following seed values: \
[12, 22, 32, 42, 52].

## Note – Handcrafted Features:
Handcrafted features have been pre-calculated and can be found at `data/hand_crafted_v3.csv` and so have some
additional readability features (`data/allreadability.pickle`). The code used to generate these features
can be found in `features.py` and `create_readability_features.py` respectively. If you wish to regenerate the features,
you will also need to install the 'textstat', 'spacy' (along with the 'en_core_web_sm' model), and 'readability' 
(use `pip install https://github.com/andreasvc/readability/tarball/master`) packages.