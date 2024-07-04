from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


dataset = pd.read_csv('datasets\\movie_reviews_dataset_bangla.csv')

# Data cleaning
nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('bengali')
all_stopwords.remove("না")

corpus = []

for i in range(0, len(dataset)):
    # Remove all characters except অ-ঔ and ক-য়
    review = re.sub('[^অ-ঔক-য়]', ' ', dataset['review'][i])
    # Convert to lowercase
    review = review.lower()
    # Split the words
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Predicting if a single review is positive or negative
new_review = 'ভালো লাগছে'
new_review = re.sub('[^অ-ঔক-য়]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
new_review = [ps.stem(word)
              for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
