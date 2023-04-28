import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import re
import warnings
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv("data.csv")

def process_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    text = " ".join(filtered_sentence)
    return text

data['Text_parsed'] = data['Text'].apply(process_text)

label_encoder = preprocessing.LabelEncoder()
data['Category_target'] = label_encoder.fit_transform(data['Category'])

X_train, X_test, y_train, y_test = train_test_split(data['Text_parsed'],
                                                    data['Category_target'],
                                                    test_size=0.2,
                                                    random_state=8)

ngram_range = (1, 2)
min_df = 10
max_df = 1
max_features = 300

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=min_df,
                        min_df=max_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

# Doc2Vec model
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
d2v_model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4, epochs=10)

# Transform text to vectors using the Doc2Vec model
def get_vectors(model, corpus, size):
    vectors = [model.infer_vector(doc.split()) for doc in corpus]
    return vectors

train_vectors = get_vectors(d2v_model, X_train, 300)
test_vectors = get_vectors(d2v_model, X_test, 300)

# Combine Tfidf vectors with Doc2Vec vectors
features_train = tfidf.fit_transform(X_train).toarray()
features_train = [features_train[i] + train_vectors[i] for i in range(len(train_vectors))]
labels_train = y_train

features_test = tfidf.transform(X_test).toarray()
features_test = [features_test[i] + test_vectors[i] for i in range(len(test_vectors))]
labels_test = y_test

LR = LogisticRegression(C=1)
LR.fit(features_train, labels_train)
model_predictions = LR.predict(features_test)
print('Accuracy : ', accuracy_score(labels_test, model_predictions))
print(classification_report(labels_test, model_predictions))

filename = 'news_classification_model.pkl'
pickle.dump(LR, open(filename, 'wb'))