#!/usr/bin/env python
# coding: utf8
"""
Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
import os
import random

import spacy
from spacy.util import minibatch, compounding

import plac
from pathlib import Path

import data_load
import app_model

TRAINED_MODEL_PATH = os.path.join('data', 'model')


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path),
)
def main(
    model=None, 
    output_dir=TRAINED_MODEL_PATH, 
    n_iter=10, 
    n_texts=2000, 
    init_tok2vec=None
    ):

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    nlp = app_model.load_model(model)
    nlp = app_model.add_category_pipe(
        nlp, 
        "textcat", 
        ["POSITIVE", "NEGATIVE"], 
        {"exclusive_classes": True, "architecture": "ensemble"}
    )

    # load the IMDB dataset
    print("Loading IMDB data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = data_load.load_training_data()
    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )
    
    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        nlp, optimizer = app_model.train_model(nlp, "textcat", ((train_texts, train_cats), (dev_texts, dev_cats)), n_iter)
 
    # test the trained model
    test_text = "This movie sucked"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)



if __name__ == "__main__":
    plac.call(main)

