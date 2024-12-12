import firebase_admin
from firebase_admin import credentials, firestore, messaging
import datetime as dt


class AlarmGenerator:
    def __init__(self, credential_path):
        """
        Initializes Firebase Admin SDK and Firestore connection.
        :param credential_path: Path to the Firebase credentials JSON file.
        """
        if not firebase_admin._apps:
            cred = credentials.Certificate(credential_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.tokens = []
        self.last_alarm = dt.datetime.now() + dt.timedelta(minutes=5)

    def fetch_tokens(self, collection_name, token_field):
        """
        Fetches FCM tokens from the specified Firestore collection.
        :param collection_name: Name of the Firestore collection.
        :param token_field: Field name where the FCM token is stored.
        """
        self.tokens = []
        collection_ref = self.db.collection(collection_name)
        docs = collection_ref.stream()

        for doc in docs:
            token = doc.to_dict().get(token_field)
            if token:
                self.tokens.append(token)

    def send_notifications(self, title, body):
        """
        Sends notifications to all fetched tokens.
        :param title: Title of the notification.
        :param body: Body of the notification.
        """
        if not self.tokens:
            print("No tokens found.")
            return

        if dt.datetime.now() < self.last_alarm + dt.timedelta(minutes=5):
            return

        for token in self.tokens:
            try:
                message = messaging.Message(
                    notification=messaging.Notification(title=title, body=body),
                    token=token,
                )
                response = messaging.send(message)
                print(f"Notification sent successfully to token {token}: {response}")
            except Exception as e:
                print(f"Error sending notification to token {token}: {e}")

    def clear_tokens(self):
        """
        Clears the list of tokens.
        """
        self.tokens = []
        print("Token list cleared.")


# Example usage
if __name__ == "__main__":
    sender = AlarmGenerator("key.json")

    # Fetch tokens from the 'teachers' collection, field 'fcmToken'
    sender.fetch_tokens("teachers", "fcmToken")

    # Send notifications if tokens are available
    sender.send_notifications(
        title="BULLYING WARNING", body="Possible bullying in the hallway"
    )