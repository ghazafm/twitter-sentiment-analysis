import pandas as pd
import pickle

def extract_date(date_string):
    date_parts = date_string.split()

    month_dict = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }

    date = date_parts[2]
    month = month_dict[date_parts[1]]
    year = date_parts[-1]

    result = f'{year}-{month}-{date}'
    return result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def remove_str_index(word, token):
    temp = [0]
    for j in word:
        if not j == temp[-1]:
            temp.append(j)
    temp.remove(0)
    temp = "".join(temp)
    return temp


def clear_double(data, token=False):
    temp = []
    data = data.split()
    for word in data:
        temp.append(remove_str_index(word, token))
    if token:
        return temp
    else:
        return " ".join(temp)


def cek_alay(word, alay):
    return alay.get(word, word)


def clear_alay(data,alay):
    words = str(data)
    words = words.split()
    cleared_words = [cek_alay(word, alay) for word in words]
    return " ".join(cleared_words)


def tokenizer(text):
    from nltk import word_tokenize
    text = word_tokenize(text)
    return text


double_meaning = [
    "jadi",
    "menjadi",
    "bapak",
    "kalau",
    "rakyat",
    "siapa",
    "apa",
    "orang",
    "bakal",
    "sama",
    "pasang",
    "jelang",
    "tahun",
    "hari",
    "bersama",
    "mau",
    "tetap",
    "buat",
    "for",
    "bukan",
    "semua",
    "terus",
    "si",
    "inilah",
    "kan",
    "tak",
    "banyak",
    "meski",
    "lebih",
    "keputusan",
    "final",
    "paling",
    "hasil",
    "umum",
    "tepat",
    "tersebut",
    "total",
    "klik",
    "capres",
    "pilih",
    "pemilihan",
    "terpilih",
    "survei",
    "survey",
    "pemilu",
    "terkait",
    "fahnoor",
    "nan",
    "calon",
    "pilpres",
    "resmi",
    "cocok",
    "politik",
    "ribuan",
    "ratusan",
    "nama",
    "maju",
    "hut",
    "dapat",
    "semoga",
    "beliau",
    "besar",
    "makin",
    "layak",
    "partai",
    "mendukung",
    "dukung",
    "dukungan",
    "gubernur",
    "masyarakat",
    "warga",
    "presiden",
    "ri",
    "inismyname",
    "pilpres",
    "nan",
    "calon",
    "indonesia",
    "survei",
    "survey",
    "pemilu",
    "aa",
    "aah",
    "aak",
    "aan",
]

name = [
    "inismyname",
    "indonesia",
    "rosiade",
    "joko",
    "jokowi",
    "widodo",
    "ridwan",
    "kamil",
    "rosiade",
    "thohir",
    "mujani",
    "erick",
    "saiful",
    "chotimah",
    "ahy",
    "bukan",
    "aniesahy",
    "ahmad",
    "pks",
    "pdip",
    "jawa",
    "puan",
    "maharani",
    "pan",
    "jateng",
    "tengah",
    "megawati",
    "ppp",
    "rasyid",
    "gerindra",
    "nasdem",
    "demokrat",
    "pkb",
    "allah",  # maaffff,
    "anis",
    "anies",
    "baswedan",
    "prabowo",
    "subianto",
    "ganjar",
    "pranowo",
    "fahnoor",
    "amien",
    "sandiaga",
    "chotimah",
    "uno",
    "aanies",
]


def clean_manual(data, token=False, tag=False):
    if tag:
        temp = []
        if token:
            for tup in data:
                if tup[0] in double_meaning or tup[0] in name:
                    continue
                temp.append(tup)
            return temp
        else:
            for tup in data:
                if tup[0] in double_meaning or tup[0] in name:
                    continue
                temp.append(tup[0])
        temp = " ".join(temp)

    else:
        data = str(data)
        data = data.split()
        data = [word for word in data if not word in double_meaning]
        data = [word for word in data if not word in name]
        temp = ' '.join(data)

    return temp


def stemming(data, stemmer, token=False, tag=False):
    last = ""
    if tag:
        temp = []
        for tup in data:
            tup_temp = stemmer.stem(tup[0])
            don = [tup_temp, tup[1]]
            if token:
                temp.append(don)
                last = temp
            else:
                temp.append(tup_temp)
                last = " ".join(temp)
    else:
        last = stemmer.stem(data)
    return last


def clear_stopwords(data, stopword, token=False,tag=False):
    if tag:
        temp = []
        last = ""
        for tup in data:
            if tup[0] in stopword:
                continue
            if token:
                temp.append(tup)
                last = temp
            else:
                temp.append(tup[0])
                last = " ".join(temp)
    else:
        data = str(data)
        last = stopword.remove(data)
    return last


def join(data):
    temp = []
    for tup in data:
        temp.append(tup[1])
    return " ".join(temp)


def removena(data):
    if len(data) == 0:
        return None
    return data

def clear_char(data):
    import re
    data = data.lower()
    data = re.sub("[^a-z]", "", x)


def clean_text(data, tag=False):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    print("clean_text")
    data["clean_text"] = data["Tweet"].apply(clear_double)
    print("no_double")
    data["no_double"] = data["clean_text"].apply(clear_double)

    alay = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
    alay = dict(zip(alay["slang"], alay["formal"]))
    print("no_alay")
    data["no_alay"] = data["no_double"].apply(clear_alay,alay=alay)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    if tag:
        stopword = StopWordRemoverFactory().create_stop_word_remover().dictionary.words
        from nltk.tag import CRFTagger
        print("pos_tag")
        ct = CRFTagger()
        ct.set_model_file("all_indo_man_tag_corpus_model.crf.tagger")
        tokenize_data = data["no_alay"].apply(tokenizer)
        data["pos"] = ct.tag_sents(tokenize_data)
        tokenize_data = data["pos"]

        print("manual")
        data["clean_manual"] = tokenize_data.apply(clean_manual)
        tokenize_data = tokenize_data.apply(clean_manual, token=True)

        print("stemming")
        data["stemmed"] = tokenize_data.apply(stemming,stemmer=stemmer,tag=True)
        tokenize_data = tokenize_data.apply(stemming,stemmer=stemmer, token=True,tag=True)

        print("stopwords")
        data["no_stopwords"] = tokenize_data.apply(clear_stopwords,stopword=stopword,tag=True)
        tokenize_data = tokenize_data.apply(clear_stopwords,stopword=stopword, token=True,tag=True)

        print("tokenize")
        data["tag"] = tokenize_data.apply(join)

        print("remove null")
        data["no_stopwords"] = data["no_stopwords"].apply(removena)

        data.dropna(inplace=True)

        data = data[
            [
                "Tweet",
                "no_double",
                "no_alay",
                "pos",
                "clean_manual",
                "stemmed",
                "no_stopwords",
                "tag"
            ]
        ]

        vectorizer = pickle.load(open(
                    "../preprocessing/Training/preprocessing/pickle/countVectorizer_tag.pickle",
                    "rb",
                )
            )

        temp = vectorizer.transform(data["tag"])
        pos_tag = pd.DataFrame(
            temp.toarray(), columns=vectorizer.get_feature_names_out()
        )
        data = data.reset_index(drop=True)
        data = pd.concat([data, pos_tag], axis=1)
    else:
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        print("manual")
        data["clean_manual"] = data["no_alay"].apply(clean_manual)

        print("stemming")
        data["stemmed"] = data["clean_manual"].apply(stemming,stemmer=stemmer)

        print("stopword")
        data["no_stopwords"] = data["no_alay"].apply(clear_stopwords,stopword=stopword)

        print("remove null")
        data.dropna(inplace=True)

        data = data[
            [
                "Tweet",
                "no_double",
                "no_alay",
                "clean_manual",
                "stemmed",
                "no_stopwords",
            ]
        ]

    return data


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def remove_str_index(word, token):
    temp = [0]
    for j in word:
        if not j == temp[-1]:
            temp.append(j)
    temp.remove(0)
    temp = "".join(temp)
    return temp


def clear_double(data, token=False):
    temp = []
    data = data.split()
    for word in data:
        temp.append(remove_str_index(word, token))
    if token:
        return temp
    else:
        return " ".join(temp)


def undersampling(x):
    x_p = x[x["label"] == "Positive"]
    x_n = x[x["label"] == "Negative"]

    x_temp = x_p.sample(x_n.label.count(), random_state=42)
    x_under = pd.concat([x_temp, x_n], axis=0, ignore_index=True)
    return x_under


def tfidf_vec(data,vectorizer):
    data = vectorizer.transform(data)
    data = data.toarray()
    data = pd.DataFrame(data, columns=vectorizer.get_feature_names_out())
    return data


# train test split
def bagi_data(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2
    )
    return x_train, x_test, y_train, y_test


def preprocessing(data, tag=False):
    data = undersampling(data)
    tag = []
    if tag:
        tag = data[data.columns[-20:]]

    X = data["no_stopwords"]
    y = data["label"]
    vectorizer = pickle.load(open("../model/tag/pickle/tfidf_tag.pickle","rb",))
    X = tfidf_vec(X,vectorizer=vectorizer)
    if tag:
        X = pd.concat([X, tag], axis=1)

    x_train, x_test, y_train, y_test = bagi_data(X, y)

    return x_train, x_test, y_train, y_test


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def all_report(x, y):
    from sklearn.metrics import classification_report
    report = classification_report(x, y)
    return report
