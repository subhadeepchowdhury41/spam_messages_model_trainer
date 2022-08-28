import pandas as pd
from nlp import text_preprocessor, normalizer, stemmer, vectorizer
from trainer import model_trainer, accuracy_checker

# reading data from csv file from kaggle
message_data = pd.read_csv("spam.csv",encoding = "latin")

# removing unwanted columns and giving rest relevant names
message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
message_data = message_data.rename(columns = {'v1':'Is Spam','v2':'Message'})

# copy the messages to a new array
message_data_copy = message_data['Message'].copy()

print("")
print("")
print("Preprocessing and applying stemmer to each messages...")

# apply text preprocessor, stemmer
message_data_copy = message_data_copy.apply(text_preprocessor.text_preprocess)
message_data_copy = message_data_copy.apply(stemmer.stemmer)

print("Creating vector of the data...")

# vectorize the data and create message matrix
matrix = vectorizer.vectorizer(message_data_copy)

print("Normalizing the length data...")

# normalize the data
matrix = normalizer.get_normalizer_matrix(message_data, matrix)

print("Trainig the model...")

# trainig the data
message_train, message_test, is_spam_train, is_spam_test = model_trainer.train_model(message_data, matrix)

print("")
print("")

# find the accuracy of the model
print("Accuracy: " + str(accuracy_checker.check_accuracy(message_train, message_test, is_spam_train, is_spam_test)))

print("")
print("")