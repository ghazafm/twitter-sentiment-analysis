# label encoder
def label_encoder(y):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(y)
    return y


# under dan over sampling
def undersampling(x, y):
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler()
    x, y = rus.fit_resample(x.values.reshape(-1, 1), y)
    x = x.flatten()
    return x, y


def oversampling(x, y):
    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler()
    x, y = ros.fit_resample(x.values.reshape(-1, 1), y)
    x = x.flatten()
    return x, y


# train test split
def bagi_data(x, y):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2
    )
    return x_train, x_test, y_train, y_test


# cetak csv
def cetak_csv(data):
    data.to_csv("name.csv")


# Vectorizing
def tfidf_vec(x_train, x_test):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    x_train = pd.DataFrame(x_train, columns=vectorizer.get_feature_names_out())
    x_test = pd.DataFrame(x_test, columns=vectorizer.get_feature_names_out())
    return x_train, x_test


def count_vec(x_train, x_test):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    return x_train, x_test


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
