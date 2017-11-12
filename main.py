import SVM
import extract_features

# Reads emails and labels data from files
def read_email_data(email_file_name, labels_file_name):
    emails_file = open(email_file_name, 'r')
    emails_text = emails_file.read()
    emails_text = emails_text.replace('\n', ' ')
    emails =  emails_text.split('--------------------------------------------------------')

    lables_file = open(labels_file_name, 'r')
    labels = [int(x) for x in lables_file.read().split('\n')]

    return (emails, labels)

# Reads the list of possible websites from a file
def read_website_list(file_name):
    file = open(file_name, 'r')
    website_list = file.read().split('\n')
    
    return website_list

# Finds which which website occurs in the email
def find_intended_websites(websites, emails):
    intended_websites = []
    for email in emails:
        found = False
        for website in websites:
            if website in email:
                intended_websites.append(website)
                found = True
                break
            
        if not found:
            intended_websites.append('Unknown')
            
    return intended_websites

# Read all data
(emails, y) = read_email_data('E-MAILS.txt', 'LABELS.txt')
(test_emails, test_labels) = read_email_data('TESTS.txt', 'TEST_LABELS.txt')
websites = read_website_list('websites.txt')

#get the website about which the email is
intended_websites = find_intended_websites(websites, emails)
 


#extract the needed features from the datasets
feature_extractor = extract_features.FeatureExtractor()

#if we laoded it form the disk, then dont train it. read the trained model data
if not feature_extractor.has_vocab:
    (train_features, feature_names) = feature_extractor.extract_email_train_features(emails)
else:
    feature_names = feature_extractor.vect.get_feature_names()
    train_features = feature_extractor.vect.training_data_features

#extract the test data
test_features = feature_extractor.extract_email_test_features(test_emails)


#classify
(clf, already_trained) = SVM.get_classifier()

if not already_trained:
    SVM.train_clf(clf, train_features, y)

train_guesses = SVM.classify(clf, train_features)
test_guesses = SVM.classify(clf, test_features)

print('training performance: ', SVM.eval_performance(train_guesses, y))
print('test performance: ', SVM.eval_performance(test_guesses, test_labels))


feature_extractor.save_vectorizer()
SVM.save_classifier_to_disk(clf)
