import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
import re
import warnings

from wordcloud import WordCloud

warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv("data.csv")


def create_wordcloud(words):
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

subset=data[data.Category=="business"]
text=subset.Text.values
words =" ".join(text)
create_wordcloud(words)

subset=data[data.Category=="entertainment"]
text=subset.Text.values
words =" ".join(text)
create_wordcloud(words)

subset=data[data.Category=="politics"]
text=subset.Text.values
words =" ".join(text)
create_wordcloud(words)

subset=data[data.Category=="sport"]
text=subset.Text.values
words =" ".join(text)
create_wordcloud(words)

subset=data[data.Category=="tech"]
text=subset.Text.values
words =" ".join(text)
create_wordcloud(words)

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

pickle.dump(tfidf,open('feature.pkl','wb'))

features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test

LR = LogisticRegression(C=1)
LR.fit(features_train, labels_train)
model_predictions = LR.predict(features_test)
print('Accuracy : ', accuracy_score(labels_test, model_predictions))
print(classification_report(labels_test, model_predictions))

filename = 'news_classification_model.pkl'
pickle.dump(LR, open(filename, 'wb'))


# def classify_text(text1, true_label):
#     # Preprocess the text
#     processed_text = process_text(text1)
#
#     # Vectorize the text using the same TfidfVectorizer as in the training
#     text_vector = tfidf.transform([processed_text]).toarray()
#
#     # Predict the category using the trained Logistic Regression model
#     category_idx = LR.predict(text_vector)[0]
#
#     # Map the category index to its label using the LabelEncoder
#     category_label = label_encoder.inverse_transform([category_idx])[0]
#
#     if category_label == true_label:
#         print("The model predicts the correct label for the given text.")
#     else:
#         print("The model predicts a different label than the true label for the given text.")
#
#     return category_label
#
#
# pred_text = ["Nepal Police Club defeated FC Khumaltar 3-1, while Friends Club salvaged a 1-1 draw with APF Football "
#              "Club in the Martyrs Memorial A Division League here today. "]
#
#
# print("prediction:", classify_text(pred_text,3))

def Accuracy():
    return round(accuracy_score(labels_test, model_predictions) * 100, 2)
