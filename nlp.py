# -*- coding: utf-8 -*-
#%% Lendo dados
import pandas as pd

#df = pd.read_csv('amazon_lab.csv', encoding='utf-8')
#df = pd.read_csv('imdb_labelled.csv', encoding='utf-8', sep='\t')
df = pd.read_csv('yelp_labelled.csv', encoding='utf-8', sep='\t')

df.head(3)
#%%
import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

X = df['review']
y = df['sentiment']

X_clean = X.apply(preprocessor)
#%%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X_tf = vectorizer.fit_transform(X_clean)

#%%
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X_tfidf = transformer.fit_transform(X_tf)
#%%
from sklearn.decomposition import TruncatedSVD
from sklearn import tree, svm, neighbors, metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
#slr = svm.SVC(kernel="linear", C=1)
#slr = tree.DecisionTreeClassifier()
trem = neighbors.KNeighborsClassifier(n_neighbors = 3)
trem2 = neighbors.KNeighborsClassifier(n_neighbors = 3)

#MATRIZ DE CONFUSÃO

#from sklearn.linear_model import LogisticRegression
#slr = LogisticRegression()
#
trem.fit(X_tf, y)

svd = TruncatedSVD(n_components=200, n_iter=7)

svd.fit(X_tf)
new_X = svd.transform(X_tf)

trem2.fit(new_X,y)
#
#y_pred = slr.predict(X_tf)
y_pred = trem.predict(X_tf)
y_pred_r = trem2.predict(new_X)

scores = cross_val_score(trem, X_tf, y_pred, cv=100)
cross_pred = cross_val_predict(trem, X_tf, y)

scores_r = cross_val_score(trem2, new_X, y_pred_r, cv=100)
cross_pred_r = cross_val_predict(trem2, new_X, y_pred_r)
from sklearn.metrics import accuracy_score
print("Treino: ")
print(accuracy_score(y_pred, y))
print("Teste: ")
print(scores.mean())

print("\nTreino Redux: ")
print(accuracy_score(y_pred_r, y))
print("Teste Redux: ")
print(scores_r.mean())

