# Transformer

This code implements the scaled dot-product self-attention mechanism from the Transformer architecture, written from scratch in NumPy. Given an input matrix `X` and three learned projection matrices `W_Q`, `W_K`, and `W_V`, it computes queries, keys, and values, then produces attention-weighted outputs using the standard `softmax(QKᵀ / √d_k) V` formula.

Building this clarified what attention actually *is* underneath the framework abstractions: a similarity score between every pair of tokens (the `QKᵀ` matrix), normalized into weights via softmax, then used to mix the value vectors. The scaling by `√d_k` matters because without it the dot products grow large and softmax saturates into near-one-hot vectors, killing gradients. Writing softmax by hand also forced me to think about the row-wise normalization that's easy to gloss over when calling a library function.

The output is a matrix where each row is a new representation of the corresponding input token, computed as a weighted blend of all token values in the sequence. Tokens that the attention mechanism deems relevant to a given position contribute more to that position's output — this is how a Transformer lets every token "look at" every other token in a single layer.
