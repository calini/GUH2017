import SVM
import extract_features
import time
from enchant.checker import SpellChecker

# Reads emails and labels data from files
def read_email_data(email_file_name, labels_file_name):
    emails_file = open(email_file_name, 'r')
    emails_text = emails_file.read()
    emails_text = emails_text.replace('\n', ' ')
    emails =  emails_text.split('--------------------------------------------------------')

    labels_file = open(labels_file_name, 'r')
    labels = [int(x) for x in labels_file.read().split('\n')]

    emails_file.close()
    labels_file.close()

    return (emails, labels)

# Reads the list of possible websites from a file
def read_website_list(file_name):
    file = open(file_name, 'r')
    website_list = file.read().split('\n')
    
    return website_list

# Creates a dictionary which maps SVM classes to their names
def create_class_dict():
    class_dict = {
            0 : 'Delivery',
            1 : 'Order tracking',
            2 : 'Product problem',
            3 : 'Refund',
            4 : 'Cancellation',
            5 : 'Payment issue',
            6 : 'Product information',
            7 : 'Website feedback',
            8 : 'Offers',
            9 : 'Nutrition advice'}
    
    return class_dict

# Finds which which website occurs in the email
def find_intended_websites(websites, emails):
    intended_websites = []
    for email in emails:
        intended_websites.append([])
        for website in websites:
            if website.lower() in email.lower():
                intended_websites[-1].append(website)
            
    return intended_websites

#uses enchant to detect and correct typos in a single block of text
def detect_and_correct_typos(spell_chck, email):
    spell_chck.set_text(email)
    for error in spell_chck:
        if error.word.lower() in error.suggest():
            spell_chck.add()
        else:
            email = email.replace(error.word, error.suggest()[0])
    return email

def classify_email(email):
    #load the training data and the website names from files
    (training_emails, y) = read_email_data('txt_lists/E-MAILS.txt', 'txt_lists/LABELS.txt')
    websites = read_website_list('txt_lists/websites.txt')

    #get the website about which the email is
    intended_websites = find_intended_websites(websites, [email])

    class_dict = create_class_dict() #a map of indeces to names of classes

    #add the websites as part of the vocabulary of the spell check
    spell_check = SpellChecker('en')
    for word in websites.split('\n'):
        spell_check.add(word)

    #check for spelling
    email = detect_and_correct_typos(spell_check, email)

    #extract the needed features from the datasets
    feature_extractor = extract_features.FeatureExtractor()

    #if we loaded it form the disk, then don't train it. read the trained model data
    if not feature_extractor.has_vocab:
        (train_features, feature_names) = feature_extractor.extract_email_train_features(training_emails)
    else:
        feature_names = feature_extractor.vect.get_feature_names()
        train_features = feature_extractor.vect.training_data_features

    #classify
    (clf, already_trained) = SVM.get_classifier()

    if not already_trained:
        SVM.train_clf(clf, train_features, y)

    #extract the email features
    features = feature_extractor.extract_email_test_features([email])

    #build the lables for the mail client
    labels = [class_dict[SVM.classify(clf, features).tolist()[0]]]

    #append the knows websites and the sentiment score of this email
    if len(intended_websites[0]) != 0:
        labels = labels + intended_websites[0]
    sentmnt_score = features[0][-1]

    #classify the priority of the email
    if sentmnt_score < -0.7:
        labels.append('High priority')
    elif sentmnt_score < 0.2:
        labels.append('Medium priority')
    else:
        labels.append('Low priority')

    #save the models if they aren't already saved
    if not already_trained:
        SVM.save_classifier_to_disk(clf)
    if not feature_extractor.has_vocab:
        feature_extractor.save_vectorizer()

    return labels


#does the same as the same as classify email but for all emails foind in the test dataset
def test_model():
    start_time = time.time()
    # Read all data
    (emails, y) = read_email_data('txt_lists/E-MAILS.txt', 'txt_lists/LABELS.txt')
    (test_emails, test_labels) = read_email_data('txt_lists/TESTS.txt', 'txt_lists/TEST_LABELS.txt')
    websites = read_website_list('txt_lists/websites.txt')

    #get the website about which the email is
    intended_websites = find_intended_websites(websites, emails)

    #a dictionary with the actual class names
    class_dict = create_class_dict()

    #check for easy typos
    spell_chck = SpellChecker('en')
    [spell_chck.add(word) for word in open('websites.txt').read().split('\n')];
    emails = [detect_and_correct_typos(spell_chck, mail) for mail in emails]
    test_emails = [detect_and_correct_typos(spell_chck, mail) for mail in test_emails]

    #extract the needed features from the datasets
    feature_extractor = extract_features.FeatureExtractor()

    #if we laoded it form the disk, then dont train it. read the trained model data
    if not feature_extractor.has_vocab:
        (train_features, ) = feature_extractor.extract_email_train_features(emails)
    else:
        #feature_names = feature_extractor.vect.get_feature_names()
        train_features = feature_extractor.vect.training_data_features

    #extract the test data
    test_features = feature_extractor.extract_email_test_features(test_emails)


    #classify
    (clf, already_trained) = SVM.get_classifier()

    if not already_trained:
        SVM.train_clf(clf, train_features, y)

    train_guesses = SVM.classify(clf, train_features)
    test_guesses = SVM.classify(clf, test_features)

    print('training performance: ', SVM.eval_performance(train_guesses, y, class_dict))
    print('test performance: ', SVM.eval_performance(test_guesses, test_labels, class_dict))


    feature_extractor.save_vectorizer()
    SVM.save_classifier_to_disk(clf)

    print("--- %s seconds ---" % (time.time() - start_time))