import SVM
import extract_features

# Reads email data from a file
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

intended_websites = find_intended_websites(websites, emails)
print(intended_websites)   

feature_extractor = extract_features.FeatureExtractor()

(vect, train_features, feature_names) = feature_extractor.extract_email_train_features(emails)
test_features = feature_extractor.extract_email_test_features(vect, test_emails)

clf = SVM.get_classifier()

SVM.train_clf(clf, train_features, y)

train_guesses = SVM.classify(clf, train_features)
test_guesses = SVM.classify(clf, test_features)

print('training performance: ', SVM.eval_performance(train_guesses, y))
print('test performance: ', SVM.eval_performance(test_guesses, test_labels))



