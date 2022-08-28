import numpy as np

def get_normalizer_matrix(message_data, message_matrix):
    message_data['length'] = message_data['Message'].apply(len)
    length = message_data['length']
    matrx = np.hstack((message_matrix.todense(),length.to_numpy()[:, None]))
    return np.asarray(matrx)