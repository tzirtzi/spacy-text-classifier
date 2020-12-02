import random
import thinc.extra.datasets



def fetch_data(limit=0):
    """Load data from the IMDB dataset."""
    train_data, _ = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    return train_data

def labelize_data(train_data):
    texts, labels = zip(*train_data)
    return texts, labels

def categorize_data(labelList):
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labelList]
    return cats

def split_train_test_data(textList, categoriesList, split=0.8):
    
    splitIndex = int(len(textList) * split)
    
    train_set = (textList[:splitIndex], categoriesList[:splitIndex])
    test_set = (textList[splitIndex:], categoriesList[splitIndex:])

    return train_set, test_set


def load_training_data(limit=0, split=0.8, categorize_data=categorize_data, labelize_data=labelize_data, fetch_data=fetch_data):
    # Partition off part of the train data for evaluation
    train_data = fetch_data(limit)
    texts, labels = labelize_data(train_data)
    cats = categorize_data(labels)
    train_set, test_set = split_train_test_data(texts, cats, split)

    return train_set, test_set
