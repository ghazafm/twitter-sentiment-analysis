import pandas as pd

def undersampling(x):
    x_p = x[x['label']=='Positive']
    x_n = x[x['label']=='Negative']

    x_temp = x_p.sample(x_n.label.count(),random_state=42)
    x_under = pd.concat([x_temp,x_n],axis=0,ignore_index=True)
    return x_under

def tfidf_vec(data,vector=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data)
    data = vectorizer.transform(data)
    pickle.dump(vectorizer, open("pickle/tag.pkl", "wb"))
    data = data.toarray()
    data = pd.DataFrame(data, columns=vectorizer.get_feature_names_out())
    if vector:
        return data,vectorizer
    else:
        return data

# train test split
def bagi_data(x, y):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2
    )
    return x_train, x_test, y_train, y_test

def preprocessing(data,tag=False,vector=True):
    data = undersampling(data)
    tag = []
    if tag:
        tag = data[data.columns[-20:]]

    X = data['no_stopwords']
    y = data['label']
    if vector:
        X,vector = tfidf_vec(X,vector=True)
    
    if tag:
        X = pd.concat([X,tag],axis=1)

    x_train, x_test, y_train, y_test = bagi_data(X,y)

    # if tag:
    #     tag_train = x_train[x_train.columns[-20:]]
    #     tag_test = x_test[x_test.columns[-20:]]
    
    #     x_train = x_train.drop(x_train[x_train.columns[-20:]],axis=1)
    #     x_test = x_test.drop(x_test[x_test.columns[-20:]],axis=1)

    #     x_train = pd.concat([x_train,tag_train],axis=1)
    #     x_test = pd.concat([x_test,tag_test],axis=1)
    # if vector:
    #     return x_train, x_test, y_train, y_test,vector
    # else:
    return x_train, x_test, y_train, y_test


# Metric Scorer
def accuracy_score(y_test, y_pred):
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def f1_score_pos(x, y):
    from sklearn.metrics import f1_score

    f1 = f1_score(x, y, pos_label="Positive")
    return f1


def f1_score_neg(x, y):
    from sklearn.metrics import f1_score

    f1 = f1_score(x, y, pos_label="Negative")
    return f1


def precission_pos(x, y):
    from sklearn.metrics import precision_score

    precision = precision_score(x, y, pos_label="Positive")
    return precision


def precission_neg(x, y):
    from sklearn.metrics import precision_score

    precision = precision_score(x, y, pos_label="Negative")
    return precision


def recall_pos(x, y):
    from sklearn.metrics import recall_score

    recall = recall_score(x, y, "Positive")
    return recall


def recall_neg(x, y):
    from sklearn.metrics import recall_score

    recall = recall_score(x, y, "Negative")
    return recall


def all_report(x, y):
    from sklearn.metrics import classification_report

    report = classification_report(x, y)
    return report
