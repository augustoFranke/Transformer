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

    att_output = np.matmul(attention_weights, input_to_value)

    return att_output

def feed_forward(X, W1, W2):
    result = np.maximum(0, np.matmul(X, W1))
    ff_output = np.matmul(result, W2) 
    return ff_output

def layer_norm(X):
    ln_output = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-6)
    return ln_output

def transformer_block(X, W1, W2, W_Q, W_K, W_V):
    X = layer_norm(X + attention(X, W_Q, W_K, W_V))
    X = layer_norm(X + feed_forward(X, W1, W2))
    return X

X = np.random.randn(4, 8)
W_Q = np.random.randn(8, 8)
W_K = np.random.randn(8, 8)
W_V = np.random.randn(8, 8)

W1 = np.random.randn(8, 32)
W2 = np.random.randn(32, 8)

print(transformer_block(X, W1, W2, W_Q, W_K, W_V))