# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

#client = language.LanguageServiceClient()

'''emails = ['I am very upset with the product',
          'The delivery was amazing', 
          'The product was terrible',
          'I am writing this to complain about the delivery',
          'Do you sell proteins?',
          ]
test_email = 'The delivery was terrible'''
class FeatureExtractor:

    #checks if there is a copy of a vectorizer on the disk and loads it
    #or creates a new one otherwise
    def __init__(self):
        self.google_api_client = language.LanguageServiceClient()
        try:
            saved_vect = open('saved_models/vectorizer', 'rb')
            self.vect = pickle.loads(saved_vect.read())
            saved_vect.close()
            self.has_vocab = True
        except IOError:
            self.vect = TfidfVectorizer(stop_words='english')
            self.has_vocab = False

    # Returns the feature matrix for a set of training emails
    def extract_email_train_features(self, emails):
        # Calculate the inverse frequency matrix
        frequency_matrix = self.vect.fit_transform(emails)
        frequency_matrix = frequency_matrix.todense()
        frequency_matrix = np.array(frequency_matrix)
        
        # Calculate the sentiments
        sentiments = [self.extract_sentiment(email) for email in emails]
        sentiments = np.array(sentiments)
        sentiments = sentiments.reshape(len(emails), 1)
        
        # Append both matrices to obtain the feature matrix
        X = np.concatenate((frequency_matrix, sentiments), axis=1)

        self.vect.training_data_features = X

        return (X, self.vect.get_feature_names())

    # Calculates sentiment of a message, uses Google Cloud Language API
    def extract_sentiment(self, message):
        # client = language.LanguageServiceClient() # Needs optimization
        
        document = types.Document(
        content=message,
        language='en',
        type=enums.Document.Type.PLAIN_TEXT)
        
        sentiment = self.google_api_client.analyze_sentiment(document)
        return sentiment.document_sentiment.score

    def extract_email_test_features(self, emails):
        frequency_vector = self.vect.transform(emails)
        frequency_vector = frequency_vector.todense()
        frequency_vector = np.array(frequency_vector)
        
        sentiments = [self.extract_sentiment(email) for email in emails]
        sentiments = np.array(sentiments)
        sentiments = sentiments.reshape(len(emails), 1)
        
        X = np.concatenate((frequency_vector, sentiments), axis=1)

        return X

    def save_vectorizer(self):
        string_vect = pickle.dumps(self.vect)
        file_to_write_to = open('saved_models/vectorizer', 'wb')
        file_to_write_to.write(string_vect)
        file_to_write_to.close()


    

'''(vect, X, feature_names) = extract_email_train_features(emails)
print(feature_names)
print(X)
print(extract_email_test_features(vect, test_email))'''
