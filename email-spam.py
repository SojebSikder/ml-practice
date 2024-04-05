from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from os import walk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Visualizing the results
import seaborn as sns

# dataset: https://www.kaggle.com/datasets/wanderfj/enron-spam/code
# Data
pathwalk = walk('./datasets/enron')
allHamData, allSpamData = [], []

for root, dirs, files in pathwalk:
    if 'ham' in str(files):
        for obj in files:
            with open(root + '/' + obj, encoding='latin1') as ip:
                allHamData.append(" ".join(ip.readlines()))
    elif 'spam' in str(files):
        for obj in files:
            with open(root + '/' + obj, encoding='latin1') as ip:
                allSpamData.append(" ".join(ip.readlines()))


# remove all redundent data
allHamData = list(set(allHamData))
allSpamData = list(set(allSpamData))

# storing it in a dataframe
hamPlusSpamData = allHamData + allSpamData
labels = ["ham"] * len(allHamData) + ["spam"] * len(allSpamData)

raw_df = pd.DataFrame({"email": hamPlusSpamData, "label": labels})


# Preprocessing
def preprocess_text(text):
    # Remove all the special characters
    text = re.sub(r'\W', ' ', text)
    # Lowercase the text
    text = text.lower()
    # remove all single characters
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    # remove single characters at the beginning
    text = re.sub(r'^[a-z]\s+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text


# Apply the preprocessing to the email column
raw_df['email'] = raw_df['email'].apply(preprocess_text)

# Tokenization
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(raw_df['email']).toarray()

# TF-IDF
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, raw_df['label'], test_size=0.2, random_state=0)

# Training the model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Evaluating the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# Predicting new email
def predict_new_email(email):
    email = preprocess_text(email)
    email = cv.transform([email]).toarray()
    email = tfidfconverter.transform(email).toarray()
    prediction = classifier.predict(email)
    return prediction


email = "You have won $1,000,000. Please click the link below to claim your prize."
print(predict_new_email(email))


# Visualizing the results
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True)
# plt.show()
