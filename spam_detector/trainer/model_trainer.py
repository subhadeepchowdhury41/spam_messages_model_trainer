from sklearn.model_selection import train_test_split

def train_model(message_data, message_matrix):
    message_train, message_test, is_spam_train, is_spam_test = train_test_split(message_matrix,
                                message_data['Is Spam'], test_size=0.3, random_state=20)
    
    return [message_train, message_test, is_spam_train, is_spam_test]