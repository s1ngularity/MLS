# Detection insults in comments
### Winner - LogisticRegression(solver='liblinear')
### AUC = 0.871002

import pandas

train_data = pandas.read_csv("train.csv")  # 3947 rows x 3 columns
test_data = pandas.read_csv("test_with_solutions.csv")  # 2647 rows x 3 columns

X_train = train_data['Comment'].values
X_test = test_data['Comment'].values

Y_train = train_data['Insult'].values
Y_test = test_data['Insult'].values

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("#number of words = %d" % len(vectorizer.get_feature_names()))

X_train = X_train.toarray()
X_test = X_test.toarray()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, Y_train)
prediction = clf.predict_proba(X_test)  # predict probabilities

from sklearn.metrics import roc_auc_score
print("#accuracy(GaussianNB) = %f" % roc_auc_score(
    Y_test, prediction[:, 1]))  # calculate area under ROC-curve
# 0.631424

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, Y_train)
prediction = clf.predict_proba(X_test)

print("#accuracy(MultinomialNB) = %f" %
      roc_auc_score(Y_test, prediction[:, 1]))  # 0.830554

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(alpha=1.85)
clf.fit(X_train, Y_train)
prediction = clf.predict_proba(X_test)

print("#accuracy(BernoulliNB) = %f" %
      roc_auc_score(Y_test, prediction[:, 1]))  # 0.863118

#from sklearn.svm import SVC
#clf = SVC(kernel='rbf', probability=True)
#clf.fit(X_train, Y_train)
#prediction = clf.predict_proba(X_test)
#
# print("#accuracy = %f" % roc_auc_score(Y_test, prediction[:, 1]))  # 0.839606

from sklearn.linear_model import LogisticRegression
# SOLVERS ‘newton-cg=0.871307’,‘lbfgs=0.871305 ’,‘liblinear=0.871002’,
# ‘sag=0.830833’, ‘saga=0.80567’
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, Y_train)
prediction = clf.predict_proba(X_test)

print("#accuracy(LogisticRegression) = %f" %
      roc_auc_score(Y_test, prediction[:, 1]))
