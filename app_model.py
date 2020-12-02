import random 
import spacy
from spacy.util import minibatch, compounding


def load_model(model=None):

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    return nlp


def add_category_pipe(nlp, categoryName, categoryLabels, 
        categoryModelConfig={"exclusive_classes": True, "architecture": "ensemble"}):

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if categoryName not in nlp.pipe_names:
        textcat = nlp.create_pipe(categoryName, config=categoryModelConfig)
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe(categoryName)
    
    for label in categoryLabels:
        textcat.add_label(label)
    
    return nlp


# def evaluate(tokenizer, textcat, texts, cats):
#     docs = (tokenizer(text) for text in texts)
#     tp = 0.0  # True positives
#     fp = 1e-8  # False positives
#     fn = 1e-8  # False negatives
#     tn = 0.0  # True negatives
#     for i, doc in enumerate(textcat.pipe(docs)):
#         gold = cats[i]
#         for label, score in doc.cats.items():
#             if label not in gold:
#                 continue
#             if label == "NEGATIVE":
#                 continue
#             if score >= 0.5 and gold[label] >= 0.5:
#                 tp += 1.0
#             elif score >= 0.5 and gold[label] < 0.5:
#                 fp += 1.0
#             elif score < 0.5 and gold[label] < 0.5:
#                 tn += 1
#             elif score < 0.5 and gold[label] >= 0.5:
#                 fn += 1
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     if (precision + recall) == 0:
#         f_score = 0.0
#     else:
#         f_score = 2 * (precision * recall) / (precision + recall)
#     return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def train_model(nlp, categoryName, data, n_iter, batch_sizes=compounding(4.0, 32.0, 1.001)):
    
    (train_texts, train_cats), (dev_texts, dev_cats) = data
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
    
    optimizer = nlp.begin_training()
    textcat = nlp.get_pipe(categoryName)

    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}".format("iter", "LOSS", "P", "R", "F"))

    for i in range(n_iter):
        losses = {}
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)

        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)

        print("{0:.0f}\t{1:.3f}".format(i, losses["textcat"]))

        # with textcat.model.use_params(optimizer.averages):
        #     # evaluate on the dev data split off in load_data()
        #     scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)

        # print(
        #     "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
        #         losses["textcat"],
        #         scores["textcat_p"],
        #         scores["textcat_r"],
        #         scores["textcat_f"],
        #     )
        # )

    return nlp, optimizer

