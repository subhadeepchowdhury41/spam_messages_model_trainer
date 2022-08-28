from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def check_accuracy(messages_train, messages_test, is_spam_train, is_spam_test):    
    Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
    Spam_model.fit(messages_train, is_spam_train)
    pred = Spam_model.predict(messages_test)
    return accuracy_score(is_spam_test, pred)