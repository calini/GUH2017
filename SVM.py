from sklearn.svm import LinearSVC

#returns a linear SVM classifier
#see default values used for the model at
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
def get_classifier():
    return LinearSVC(random_state=int)

#trains the given classifier with the provided data
def train_clf(clf, feature_values, original_labels):
    clf.fit(feature_values, original_labels)


#takes a classifier, the data; returns the guessed labels
def classify(clf, feature_values):
    return clf.predict(feature_values)