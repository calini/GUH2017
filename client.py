import httplib2
import os

import base64
import email
import random
from apiclient import discovery
from apiclient import errors
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/gmail-python-quickstart.json
SCOPES = 'https://mail.google.com/'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Gmail API Python Quickstart'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials


def getMessageBody(service, user_id, msg_id):
    try:
        message = service.users().messages().get(userId=user_id, id=msg_id, format='raw').execute()
        msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
        mime_msg = email.message_from_string(msg_str)
        messageMainType = mime_msg.get_content_maintype()
        if messageMainType == 'multipart':
                for part in mime_msg.get_payload():
                        if part.get_content_maintype() == 'text':
                                return part.get_payload()
                return ""
        elif messageMainType == 'text':
                return mime_msg.get_payload()
    except errors.HttpError as error:
            print("An error occurred: %s" % error)

def classify(emailBody):
    categories = [
    "ADVICE",
    "CANCELLATION",
    "DELIVERY",
    "GENERAL FEEDBACK ",
    "OFFERS",
    "ORDER TRACKING",
    "PAYMENT ISSUE",
    "PRODUCT INFORMATION",
    "PRODUCT PROBLEM",
    "REFUND"]
    return [random.choice(categories)]

def categorise(service, classification, msg_id):
    try:
        '''
        labels = service.users().labels().list(userId='me').execute().get('labels', [])
        # Extract present labels
        labelNames = [labels[n]['name'] for n in range(len(labels))]
        '''

        msg_labels = {'removeLabelIds': ["INBOX"], 'addLabelIds': getLabelIds(service, classification)}

        service.users().messages().modify(id=msg_id, userId='me', body=msg_labels).execute()

    except errors.HttpError as error:
        print("An error occurred: %s" % error)

def getLabelIds(service, labels_in):
    result = []
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    for label_in in labels_in:
        for label in labels:
            if label['name'] == label_in:
                result.append(label['id'])
                break
    return result

def main():
    # Connect to the service
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)

    try:
        # Get the list of email id's from inbox to categorise 
        results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
        messages = results.get('messages', [])

        if not messages:
            print('No messages found in inbox.')
        else:
            print('Messages:')
            for message in messages:
                classification = classify(getMessageBody(service=service, user_id='me', msg_id=message['id']))
                print("Email classified as: " + ', '.join(classification))
                categorise(service, classification, message['id'])


    except errors.HttpError as error:
        print("An error occured: " + error)


if __name__ == '__main__':
    main()