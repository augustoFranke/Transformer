import numpy as np

def softmax(e):
    e = np.exp(e)
    e_sum = np.sum(e, 1)
    return e/e_sum.reshape(-1, 1)

def attention(X, W_Q, W_K, W_V):

    input_to_query = np.matmul(X, W_Q)
    input_to_key = np.matmul(X, W_K)
    input_to_value = np.matmul(X, W_V)

    Q_K = np.matmul(input_to_query, input_to_key.T)
    scaled_down_Q_K = np.divide(Q_K, np.sqrt(8))

    attention_weights = softmax(scaled_down_Q_K)

    output = np.matmul(attention_weights, input_to_value)

    return output