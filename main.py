import SVM
import extract_features
def read_input(email_file_name, labels_file_name):
    emails_file = open(email_file_name, 'r')
    emails_text = emails_file.read()
    emails_text = emails_text.replace('\n', ' ')
    emails =  emails_text.split('--------------------------------------------------------')

    lables_file = open(labels_file_name, 'r')
    labels = [int(x) for x in lables_file.read().split('\n')]

    return (emails, labels)

(emails, y) = read_input('E-MAILS.txt', 'LABELS.txt')
(test_emails, test_labels) = read_input('TESTS.txt', 'TEST_LABELS.txt')

feature_extractor = extract_features.FeatureExtractor()

(vect, train_features, feature_names) = feature_extractor.extract_email_train_features(emails)
test_features = feature_extractor.extract_email_test_features(vect, test_emails)

clf = SVM.get_classifier()

SVM.train_clf(clf, train_features, y)

train_guesses = SVM.classify(clf, train_features)
test_guesses = SVM.classify(clf, test_features)

print 'training performance: ', SVM.eval_performance(train_guesses, y)
print 'test performance: ', SVM.eval_performance(test_guesses, test_labels)



