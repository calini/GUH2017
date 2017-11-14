from sklearn.svm import LinearSVC
import pickle

#returns a linear SVM classifier
#see default values used for the model at
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html


#trains the given classifier with the provided data
def train_clf(clf, feature_values, original_labels):
    clf.fit(feature_values, original_labels)


#takes a classifier, the data; returns the guessed labels
def classify(clf, feature_values):
    return clf.predict(feature_values)

#returns a classfier if one was already saved on the disk or a new one
#also returns True if the classfier was read form disk, false if its new
def get_classifier():
    try:
        saved_classfier = open('saved_models/classifier', 'rb')
        clf = pickle.loads(saved_classfier.read())
        saved_classfier.close()
        return clf, True
    except IOError:
        return LinearSVC(random_state=0), False


#returns an object from the saved classfier string
def save_classifier_to_disk(clf):
    string_clf = pickle.dumps(clf)
    file_to_write_to = open('saved_models/classifier', 'wb')
    file_to_write_to.write(string_clf)
    file_to_write_to.close()

def eval_performance(guessed_labels, actual_labels, class_names):
    correct = 0.0

    for i in range(0, len(guessed_labels)):
        if guessed_labels[i] == actual_labels[i]:
            correct += 1
        else:
            print('#', i, 'guessed: ', class_names[guessed_labels[i]], ', actual: ', class_names[actual_labels[i]]) 

    return correct/len(guessed_labels)