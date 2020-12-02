import os
from pathlib import Path

import plac
import spacy 

TRAINED_MODEL_PATH = os.path.join('data', 'model')

@plac.annotations(
    text=("Optional output directory", "option", "t", str),
    model_path=("Optional output directory", "option", "o", Path)
)
def main(
    text,
    model_path=TRAINED_MODEL_PATH
    ):

    nlp = spacy.load(model_path)
    doc = nlp(text)
    print(text, doc.cats)


if __name__ == "__main__":
    plac.call(main)

# python app.py -t="this movie was terrific!!"