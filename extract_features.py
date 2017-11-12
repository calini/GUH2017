# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#client = language.LanguageServiceClient()

emails = ['I am very upset with the product',
          'The delivery was amazing', 
          'The product was terrible',
          'I am writing this to complain about the delivery',
          'Do you sell proteins?',
          ]
test_email = 'The delivery was terrible'

# Returns the feature matrix for a set of training emails
def extract_email_train_features(emails):
    # Calculate the inverse frequency matrix
    vect = TfidfVectorizer(stop_words='english')
    frequency_matrix = vect.fit_transform(emails)
    frequency_matrix = frequency_matrix.todense()
    frequency_matrix = np.array(frequency_matrix)
    
    # Calculate the sentiments
    sentiments = [extract_sentiment(email) for email in emails]
    sentiments = np.array(sentiments)
    sentiments = sentiments.reshape(len(emails), 1)
    
    # Append both matrices to obtain the feature matrix
    X = np.concatenate((frequency_matrix, sentiments), axis=1)

    return (vect, X, vect.get_feature_names())

# Calculates sentiment of a message, uses Google Cloud Language API
def extract_sentiment(message):
    client = language.LanguageServiceClient() # Needs optimization
    
    document = types.Document(
    content=message,
    language='en',
    type=enums.Document.Type.PLAIN_TEXT)
    
    sentiment = client.analyze_sentiment(document)
    return sentiment.document_sentiment.score

def extract_email_test_features(vect, email):
    frequency_vector = vect.transform([email])
    frequency_vector = frequency_vector.todense()
    frequency_vector = np.array(frequency_vector)
    
    sentiment = extract_sentiment(email)
    sentiment = np.array(sentiment)
    sentiment = sentiment.reshape(-1, 1)
    
    X = np.concatenate((frequency_vector, sentiment), axis=1)
    return X
    

(vect, X, feature_names) = extract_email_train_features(emails)
print(feature_names)
print(X)
print(extract_email_test_features(vect, test_email))
