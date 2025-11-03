# Attention in Transformers: Concepts and Code in PyTorch

Welcome to the companion repository for **[Attention in Transformers: Concepts and Code in PyTorch](https://www.deeplearning.ai/short-courses/attention-in-transformers-concepts-and-code-in-pytorch/)** a course by **[DeepLearning.AI](https://www.deeplearning.ai/)** in collaboration with **[StatQuest](https://statquest.org/)**.

This repository contains notes, concepts, and code implementations based on the course material.

## Introduction

The attention mechanism was a breakthrough that led to Transformers, the architecture powering modern Large Language Models (LLMs) like ChatGPT, BERT, and T5. The 2017 paper, **[Attention is All You Need](https://arxiv.org/abs/1706.03762)**, revolutionized the field by proposing a scalable design that moves beyond the sequential limitations of recurrent networks.

This course teaches you how this foundational architecture works, improving your intuition for building reliable, functional, and scalable AI applications.

### What You'll Learn

  * The **evolution of the attention mechanism** and the problems it solved.
  * The relationships between **word embeddings**, **positional embeddings**, and **attention**.
  * The concepts of **Query (Q), Key (K), and Value (V)** matrices: how to produce them and how to use them.
  * The detailed **mathematics of self-attention** and **masked self-attention**.
  * The critical difference between **self-attention** (for context-aware embeddings in the encoder) and **masked self-attention** (for generative outputs in the decoder).
  * The complete **encoder-decoder architecture**, including **cross-attention** and **multi-head attention**.
  * How to **code a complete attention module from scratch in PyTorch**, implementing self, masked, and multi-head attention.

## Course Topics

This repository is structured around the core topics of the course. You can expand each section below for a detailed, deep-dive explanation of the concepts and mathematics.

<details>
<summary><strong>1. The Main Ideas Behind Transformers and Attention</strong></summary>

### The Pre-Transformer World: RNNs and Their Limits

Before the "Attention is All You Need" paper in 2017, the state-of-the-art for sequence-processing tasks like machine translation or text summarization was dominated by **Recurrent Neural Networks (RNNs)**. Architectures like **Long Short-Term Memory (LSTMs)** and **Gated Recurrent Units (GRUs)** were the standard.

The core idea of an RNN is to process a sequence *sequentially*. It takes the first word (embedding), processes it, and produces a "hidden state." Then, it takes the *second* word and the *previous* hidden state, processes them together, and produces a *new* hidden state. This continues for the entire sequence, with the final hidden state (or a collection of them) theoretically encoding the meaning of the entire sequence.

This approach had two fundamental problems:

1.  **The Long-Range Dependency Problem:** As the sequence gets longer, information from the first few words gets "diluted" or "forgotten." The hidden state has to compress the *entire* past into a single fixed-size vector. By the time the RNN processes the 50th word, the specific details of the 1st word are often lost. LSTMs and GRUs introduced "gates" (like a "forget gate") to help manage this, but the problem still persisted for very long sequences. This is also related to the "vanishing gradient" problem, where the gradients from the end of the sequence become too small to update the network weights at the beginning of the sequence.

2.  **The Sequential Bottleneck:** The process is inherently sequential. You *cannot* process the 10th word until you have processed the 9th word. In an era of parallel computing with powerful GPUs, this was a massive computational bottleneck. You couldn't just throw more hardware at the problem to make it faster; the algorithm itself prevented parallelization across the time (sequence) dimension.

### The First Spark: Attention with RNNs (Bahdanau & Luong)

The first "attention" mechanisms (circa 2014-2015) were actually designed to *fix* the RNN, not replace it. In a "seq2seq" (sequence-to-sequence) model for translation, an "encoder" RNN would read the entire source sentence (e.g., in English) and compress it into a single context vector. A "decoder" RNN would then take this one vector and try to generate the entire target sentence (e.g., in French).

The problem was that the decoder only had *one* vector to work with. This was a huge information bottleneck.

**Attention** (specifically, Bahdanau-style "additive" attention) proposed a new idea:

  * Don't force the encoder to create one context vector. Let it produce a *sequence* of hidden states, one for each input word.
  * At *each step* of the translation, the decoder would "look back" at *all* of the encoder's hidden states.
  * It would *learn* a set of "attention weights" that told it how much "attention" to pay to each input word when generating the *current* output word.
  * For example, when generating the French word "maison," the decoder would learn to pay high attention to the English input word "house."

This worked brilliantly. It solved the information bottleneck and provided a clear "alignment" between input and output. However, it was still built on RNNs. It was still sequential and slow.

### The Revolution: "Attention is All You... Need"

The 2017 paper by Vaswani et al. made a radical proposal: **What if we throw away the RNNs entirely and *only* use the attention mechanism?**

This seemed impossible at first. How do you process a sequence without a recurrent loop? The answer was **Self-Attention**.

**Main Idea 1: Self-Attention (Intra-Attention)**
Instead of attention between a decoder and an encoder ("cross-attention"), **self-attention** allows tokens *within the same sequence* to look at each other.

To understand the word "bank" in the sentence "I sat on the river **bank**," you need to look at the word "river." To understand "bank" in "I went to the **bank** to deposit a check," you need to look at "deposit" or "check."

Self-attention is a mechanism that, for *every* word in a sentence, calculates a score (an attention weight) for *every other word* in that same sentence. It then creates a new, "contextualized" embedding for that word by taking a weighted average of all word embeddings in the sentence, where the weights are the attention scores.

The key result:

  * **No Long-Range Dependency Problem:** Every word can *directly* look at every other word, regardless of distance. The path between "I" (word 1) and "check" (word 7) is of length 1, not length 6 (as in an RNN).
  * **Full Parallelization:** The calculation for *each word* can be done *simultaneously*. The entire process is just a series of matrix multiplications, which GPUs are exceptionally good at. This solved the sequential bottleneck.

**Main Idea 2: Positional Embeddings**
If you just throw all the words into a "bag" and have them all interact at once (as self-attention does), you lose one crucial piece of information: **word order**. The sentences "The dog bit the man" and "The man bit the dog" have the same words, but self-attention, on its own, would see them as identical.

The solution is **Positional Embeddings**. Before the words are fed into the first attention layer, a vector representing the *position* of the word (e.g., 1st, 2nd, 3rd) is *added* to its word embedding.

  * `Input Embedding = Word Embedding + Positional Embedding`

This "injects" information about the sequence order into the model. The model can then learn to use this information. (The original paper used fixed sine and cosine functions to create these, but modern models like BERT often just *learn* the positional embeddings just like they learn word embeddings).

**Main Idea 3: The Encoder-Decoder Architecture**
The paper proposed a specific architecture for translation:

  * **The Encoder Stack:** A stack of $N$ identical layers (e.g., $N=6$). Its job is to read the input sequence (e.g., "the house") and produce a set of contextualized embeddings that represent its *meaning*. Each encoder layer contains:
    1.  A **Multi-Head Self-Attention** layer (to understand the *input* sentence).
    2.  A **Feed-Forward Neural Network** (a simple 2-layer network) to add computational depth.
  * **The Decoder Stack:** A stack of $N$ identical layers. Its job is to take the encoder's output and generate the target sequence (e.g., "la maison"). Each decoder layer contains *three* sub-layers:
    1.  A **Masked Multi-Head Self-Attention** layer (to look at what the decoder has *already* generated).
    2.  A **Multi-Head Cross-Attention** layer (to look at the *encoder's output*—this is where it "reads" the input sentence).
    3.  A **Feed-Forward Neural Network**.

This architecture, based *only* on attention and feed-forward layers, became the "Transformer." It was more parallelizable, could handle longer dependencies, and trained faster, leading to a new state-of-the-art in machine translation and, eventually, in almost all of NLP.

</details>

<details>
<summary><strong>2. The Matrix Math for Calculating Self-Attention</strong></summary>

### The Core Formula

At the heart of the Transformer is the "Scaled Dot-Product Attention" formula. It looks intimidating, but it's just a series of simple steps.

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$Let's break this down, one piece at a time.

* **$Q$**: Query matrix
* **$K$**: Key matrix
* **$V$**: Value matrix
* **$d_k$**: The dimension of the Key vectors.

### The Intuition: A Database Retrieval System

Think of self-attention as a fast, "soft" database retrieval system.

Imagine you have a database (your *sequence* of words). For each word (database entry), you have three pieces of information:

1.  **Value ($V$)**: The *actual content* of the word. The information it holds. "I am a vector for the word 'river'."
2.  **Key ($K$)**: A *label* or *description* of the Value. "I am a 'geography' and 'water' related concept." This is the "searchable" part.
3.  **Query ($Q$)**: A *question* you are asking the database. "I am looking for a 'geography' concept to help define me."

**Self-Attention** is a special case where *every word in the sequence is simultaneously a Query, a Key, and a Value.*

Each word (as a **Query**) "broadcasts" its question to all other words (as **Keys**).

* "I am the word 'bank' (Query), and I am looking for context."
* It compares its Query to the Key of 'river': "My query ('bank') matches 'river's Key ('geography') very well\!" -\> **High Score**
* It compares its Query to the Key of 'I': "My query ('bank') does not match 'I's Key ('pronoun') very well." -\> **Low Score**

After getting a *score* for every other word, the 'bank' Query then says: "Okay, I will now create my new, contextualized representation. I will take 1% of the **Value** of 'I', 2% of the **Value** of 'sat', 4% of 'on', 3% of 'the', and 90% of the **Value** of 'river'."

The result is a new vector for 'bank' that is "90% river-bank" and not "1% 'I'-bank". This is a **contextualized embedding**.

### The Matrix Math (The Step-by-Step "How")

Let's assume we have an input sequence of $L$ words, and each word is represented by an embedding vector of size $d\_{model}$. Our input is a matrix $X$ of shape $[L, d\_{model}]$.

**Step 1: Create $Q$, $K$, and $V$**

The $Q$, $K$, and $V$ vectors are not the embeddings themselves. They are *projections* of the embeddings. We create three new weight matrices, $W^Q$, $W^K$, and $W^V$, which are *learned* during training. Each has a shape of $[d\_{model}, d_k]$ (for simplicity, let's assume $d_k = d\_{model}$).

* **Query Matrix**: $Q = X * W^Q$
* **Key Matrix**: $K = X * W^K$
* **Value Matrix**: $V = X * W^V$

Since $X$ is $[L, d\_{model}]$ and $W^Q$ is $[d\_{model}, d_k]$, the resulting $Q$ matrix is $[L, d_k]$.
Similarly, $K$ is $[L, d_k]$ and $V$ is $[L, d_v]$ (often $d_k = d_v$).

*Why do this?* These $W$ matrices allow the model to *learn* the optimal "space" to project the embeddings into for querying, key-matching, and providing values. It gives the model flexibility. The word 'bank' might use one part of its embedding vector to *query* (e.g., "am I a financial or geographical concept?") and another part to act as a *key* (e.g., "I am a noun").

**Step 2: Calculate Scores ($QK^T$)**

This is the core of the "dot-product" attention. We want to see how much each Query $Q_i$ (for word $i$) "matches" each Key $K_j$ (for word $j$). A dot product is a measure of similarity.

We matrix-multiply $Q$ by the *transpose* of $K$.

* $Q$ shape: $[L, d_k]$
* $K^T$ (K transposed) shape: $[d_k, L]$
* **Result (Scores Matrix) shape**: $[L, L]$

Let's call this `Scores`. `Scores[i, j]` holds a single number representing the "match" or "affinity" between word $i$ (as a Query) and word $j$ (as a Key).

**Step 3: Scale ($\sqrt{d_k}$)**

This is the $/\sqrt{d_k}$ part.

* **Problem:** As the dimension $d_k$ gets large (e.g., 64 or 512), the dot products in the `Scores` matrix can become very large.
* **Why is this bad?** These large numbers are fed into a `softmax` function (Step 4). Softmax is sensitive to large inputs. If one input is `10` and another is `1000`, the softmax will output `[~0, ~1]`. This is called "saturation."
* **Consequence:** The gradients become vanishingly small, and the model stops learning.
* **Solution:** We "scale" the scores down by dividing them all by the square root of the dimension, $\sqrt{d_k}$. This keeps the variance of the scores at $\approx 1$ and ensures the softmax function stays in a "healthy" region with good gradients.

So, `ScaledScores` = `Scores` / $\sqrt{d_k}$. The shape is still $[L, L]$.

**Step 4: Softmax**

Now we have a matrix of "scaled affinity scores," but they are just numbers (positive, negative, big, small). We want to turn them into a *distribution*—a set of weights that sum to 1. This is exactly what the `softmax` function does.

We apply `softmax` *row-wise* to the `ScaledScores` matrix.

`AttentionWeights` = $\text{softmax}(\text{ScaledScores})$

* Shape is still $[L, L]$.
* `AttentionWeights[i, j]` now holds a positive number between 0 and 1.
* For a given row $i$, the sum of all $j$ columns is 1 ($\sum_j \text{AttentionWeights}[i, j] = 1$).
* `AttentionWeights[i, j]` can be interpreted as: "When calculating the new representation for word $i$, what *percentage* of 'attention' should I pay to the **Value** of word $j$?"

**Step 5: Apply to Values ($...V$)**

We have the "recipe" (the `AttentionWeights`). Now we just apply it to the "ingredients" (the `V` matrix).

We matrix-multiply our `AttentionWeights` by the `V` matrix.

* `AttentionWeights` shape: $[L, L]$
* `V` shape: $[L, d_v]$
* **Result (Output $Z$) shape**: $[L, d_v]$ (which is typically $[L, d\_{model}]$).

Let's analyze what $Z_i$ (the $i$-th row of the output $Z$) is:
$Z_i = \sum_j (\text{AttentionWeights}[i, j] * V_j)$

The new, contextualized vector for word $i$ ($Z_i$) is a **weighted sum** of *all* the Value vectors ($V_j$) in the entire sequence. The weights are precisely the attention scores we just calculated.

If `AttentionWeights[2, 5]` (row 2, col 5) is 0.9, then the new vector for word 2 ($Z_2$) will be composed 90% of the Value vector from word 5 ($V_5$).

This completes the calculation. We started with an input $X$ (shape $[L, d\_{model}]$) and produced an output $Z$ (shape $[L, d\_{model}]$) where every vector $Z_i$ is now "context-aware," having absorbed information from all other tokens in the sequence based on the learned attention patterns.

</details>

<details>
<summary><strong>3. Coding Self-Attention in PyTorch</strong></summary>

### From Math to Code

Let's translate the mathematical formula $Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ into a PyTorch `nn.Module`.


We will build a simple, single-head self-attention module. We'll handle multi-head attention later.

### The Code: A `SelfAttention` Class

Here is a full, annotated implementation.

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
"""
A simple single-head self-attention module.
"""

def __init__(self, embed_size, d_k=None):
"""
Args:
embed_size (int): The dimensionality of the input embedding (d_model).
d_k (int, optional): The dimensionality of Query, Key, and Value.
Defaults to embed_size.
"""
super(SelfAttention, self).__init__()

self.embed_size = embed_size

# If d_k is not specified, default to embed_size
if d_k is None:
d_k = embed_size

self.d_k = d_k

# In a single-head setup, d_q = d_k = d_v. We use d_k for all.
# We need the learnable weight matrices W^Q, W^K, W^V
# nn.Linear(in_features, out_features)

# self.query_linear is our W^Q
self.query_linear = nn.Linear(embed_size, d_k, bias=False)

# self.key_linear is our W^K
self.key_linear = nn.Linear(embed_size, d_k, bias=False)

# self.value_linear is our W^V
self.value_linear = nn.Linear(embed_size, d_k, bias=False)

# We also store the scaling factor
self.scale_factor = math.sqrt(self.d_k)

def forward(self, x_in):
"""
The forward pass of the self-attention layer.

Args:
x_in (torch.Tensor): The input tensor.
Shape: (batch_size, seq_len, embed_size)

Returns:
torch.Tensor: The output tensor (Z).
Shape: (batch_size, seq_len, d_k)
torch.Tensor: The attention weights.
Shape: (batch_size, seq_len, seq_len)
"""

# For self-attention, Q, K, and V all come from the same input, x_in.
# x_in shape: (batch_size, seq_len, embed_size)

# 1. Project x_in to Q, K, V
# Q = X * W^Q
# K = X * W^K
# V = X * W^V

Q = self.query_linear(x_in)
K = self.key_linear(x_in)
V = self.value_linear(x_in)

# Q, K, V shape: (batch_size, seq_len, d_k)

# 2. Calculate Scores (QK^T)
# We need to multiply Q by the transpose of K.
# Q shape: (batch, seq_len, d_k)
# K.transpose(-2, -1) shape: (batch, d_k, seq_len)
# torch.matmul handles the batch matrix multiplication (BMM) for us.

# scores shape: (batch, seq_len, seq_len)
scores = torch.matmul(Q, K.transpose(-2, -1))

# 3. Scale
# Divide by sqrt(d_k)
scaled_scores = scores / self.scale_factor

# 4. Softmax
# Apply softmax along the *last* dimension (dim=-1).
# This makes each row of the (seq_len, seq_len) matrix sum to 1.

# attention_weights shape: (batch, seq_len, seq_len)
attention_weights = torch.softmax(scaled_scores, dim=-1)

# 5. Apply to Values (matmul with V)
# attention_weights shape: (batch, seq_len, seq_len)
# V shape: (batch, seq_len, d_k)

# output (Z) shape: (batch, seq_len, d_k)
output = torch.matmul(attention_weights, V)

return output, attention_weights
```

### Dissecting the Code

* **`__init__(self, embed_size, d_k=None)`**:

* We inherit from `nn.Module`, the base class for all PyTorch models. This gives us `self.parameters()`, `.to(device)`, etc.
* `super()` must be called first.
* `embed_size` is $d\_{model}$, the dimension of the *incoming* word embeddings (e.g., 512).
* `d_k` is the dimension $d_k$ (and $d_q$, $d_v$). In single-head attention, this is often just `embed_size`. In multi-head, it will be `embed_size / num_heads`.
* `self.query_linear = nn.Linear(embed_size, d_k, bias=False)`: This *is* the $W^Q$ weight matrix. `nn.Linear` is a PyTorch layer that performs a matrix multiplication ($xW^T + b$). By setting `bias=False`, we are *only* creating the $W$ matrix. PyTorch will initialize these weights for us, and they will be "registered" as learnable parameters of our module.
* `self.scale_factor = math.sqrt(self.d_k)`: We pre-calculate $\sqrt{d_k}$ so we don't have to do it in every forward pass.

* **`forward(self, x_in)`**:

* This is where the computation happens. It takes the input `x_in`, which we assume has the shape `(batch_size, seq_len, embed_size)`. `batch_size` is for processing multiple sentences at once, `seq_len` is the number of tokens (e.g., 10 words), and `embed_size` is the dimension of each token (e.g., 512).
* `Q = self.query_linear(x_in)`: This line performs $XW^Q$. PyTorch is smart enough to apply the *same* linear layer to *every* token in the sequence (the `seq_len` dimension) and for *every* sentence in the batch (the `batch_size` dimension).
* `K.transpose(-2, -1)`: This is a crucial step. The `K` tensor has shape `(batch, seq_len, d_k)`. To multiply `(batch, seq_len, d_k)` with `(batch, seq_len, d_k)`, we need to transpose the *last two dimensions* of $K$. This gives us `(batch, d_k, seq_len)`. The `batch` dimension is left alone.
* `torch.matmul(Q, K.transpose(-2, -1))`: PyTorch's `matmul` function understands Batch Matrix Multiplication (BMM). It sees `(B, N, M)` and `(B, M, P)` and correctly produces `(B, N, P)`. In our case, $B=\text{batch\_size}$, $N=\text{seq\_len}$, $M=d\_k$, and $P=\text{seq\_len}$. The result is `(batch_size, seq_len, seq_len)`.
* `torch.softmax(scaled_scores, dim=-1)`: This is the row-wise softmax. We apply it on `dim=-1` (the last dimension). For our `(batch, seq_len, seq_len)` matrix, this means it's applied to the *last* `seq_len` dimension. For each row $i$, the scores $\text{score}[i, j]$ (for all $j$) are put through softmax.
* `output = torch.matmul(attention_weights, V)`: This is the final step. We multiply `(batch, seq_L, seq_L)` by `(batch, seq_L, d_k)`. (Here `seq_L` is `seq_len`). The `matmul` logic works again, resulting in `(batch, seq_L, d_k)`, which is exactly what we want.
* `return output, attention_weights`: We return the final contextualized embeddings (`output`) and also the `attention_weights`. Returning the weights is optional but *extremely* useful for debugging and visualization, as it lets you *see* what the model is "looking at."

### Example Usage

```python
# --- Example ---
# B = Batch Size (2 sentences)
# L = Sequence Length (10 words)
# E = Embedding Size (512 dimensions)
B, L, E = 2, 10, 512

# Create a sample input (e.g., from an embedding layer)
# Requires_grad=True is not needed here as we're not training
sample_input = torch.rand(B, L, E)

# --- Instantiate the module ---
# We'll use d_k = embed_size = 512
attention_layer = SelfAttention(embed_size=E, d_k=E)

# --- Run the forward pass ---
try:
output, weights = attention_layer(sample_input)

print("--- Input Shape ---")
print(sample_input.shape)

print("\n--- Output Shape ---")
print(output.shape)

print("\n--- Attention Weights Shape ---")
print(weights.shape)

print("\n--- Example: Weights for first sentence (sum of 1st row) ---")
# The weights for the 1st word's (row 0) query
# against all 10 key-words. Should sum to 1.
print(weights[0, 0, :].sum().item())

except Exception as e:
print(f"An error occurred: {e}")

# --- Expected Output ---
# --- Input Shape ---
# torch.Size([2, 10, 512])
#
# --- Output Shape ---
# torch.Size([2, 10, 512])
#
# --- Attention Weights Shape ---
# torch.Size([2, 10, 10])
#
# --- Example: Weights for first sentence (sum of 1st row) ---
# 1.0
```
</details>

<details>
<summary><strong>4. Self-Attention vs Masked Self-Attention</strong></summary>

### The Core Difference: Information Flow

The difference between Self-Attention and Masked Self-Attention is simple but has profound implications. It all comes down to one question: **What information is a token allowed to "see"?**

* **Self-Attention (Bi-directional):** A token at position $i$ can look at *all* tokens in the sequence, from position $0$ to position $L-1$. It can look to its **left** (past) and to its **right** (future).
* **Masked Self-Attention (Causal / Auto-regressive):** A token at position $i$ can *only* look at tokens from position $0$ up to (and including) position $i$. It can look to its **left** (past) but is *masked* from seeing its **right** (future).

### 1. Self-Attention

* **Analogy:** Reading a difficult sentence in a book. You read the whole sentence, then go back to a confusing word in the middle. To understand it, you use the context from *both* the words before it and the words *after* it.
* **How it Works:** The `AttentionWeights` matrix (shape `[seq_len, seq_len]`) is *fully* populated. A query from word $i$ receives scores from keys at *all* positions $j$.
```
# A hypothetical AttentionWeights matrix
# (row = Query, col = Key)
# Word 2 (row 2) can attend to all words (cols 0-4).
# This matrix is DENSE.
[0.7, 0.1, 0.1, 0.0, 0.1]  # Word 0
[0.1, 0.6, 0.2, 0.1, 0.0]  # Word 1
[0.1, 0.2, 0.5, 0.1, 0.1]  # Word 2
[0.0, 0.1, 0.1, 0.7, 0.1]  # Word 3
[0.2, 0.0, 0.1, 0.1, 0.6]  # Word 4
```
* **Use Case: Encoders (e.g., BERT)**
* The **goal of an encoder** is to build the richest, most deeply contextualized *representation* of an input.
* To understand the "it" in "The animal didn't cross the street because **it** was too tired," you *must* look at "animal" (past) and "tired" (future, relative to "it").
* BERT (Bidirectional Encoder Representations from Transformers) is "bidirectional" *precisely* because it uses standard Self-Attention. Its pre-training task (Masked Language Modeling) involves *guessing* a masked word based on *both* left and right context.
* **Summary:** Use Self-Attention when your goal is **understanding** or **representation** of a *complete* input sequence.

### 2. Masked Self-Attention

* **Analogy:** Writing a novel. When you are on page 10, deciding what word to write *next*, you can only re-read pages 1-9. You *cannot* look ahead to page 11 (because you haven't written it yet).
* **How it Works:** We apply a "look-ahead mask" or "causal mask" to the `ScaledScores` matrix *before* the softmax step. The mask forces the model to ignore any "future" tokens.
* We create a mask where all positions "to the right" (future) are set to $-\infty$.
* When $softmax(x - \infty)$ is calculated, $e^{-\infty}$ becomes $0$.
* This "zeroes out" the attention weights for all future tokens.
<!-- end list -->
```
# ScaledScores matrix (before mask)
[[ 1.2,  0.8,  2.1],
[ 0.9,  1.1,  0.5],
[ 0.4,  1.3,  0.7]]

# Mask (0 = keep, -inf = mask)
[[ 0, -inf, -inf],
[ 0,    0, -inf],
[ 0,    0,    0]]

# MaskedScores = ScaledScores + Mask
[[ 1.2, -inf, -inf],
[ 0.9,  1.1, -inf],
[ 0.4,  1.3,  0.7]]

# AttentionWeights = softmax(MaskedScores)
# This is a LOWER-TRIANGULAR matrix.
[[ 1.0,  0.0,  0.0]]  # Word 0 only sees word 0
[[ 0.4,  0.6,  0.0]]  # Word 1 only sees words 0, 1
[[ 0.2,  0.5,  0.3]]  # Word 2 sees words 0, 1, 2
```
* **Use Case: Decoders (e.g., GPT, LLM-based Chatbots)**
* The **goal of a decoder** is to *generate* text, one token at a time. This is an **auto-regressive** process.
* To generate `P("world" | "hello")`, the model must only be *given* "hello".
* **Training vs. Inference:** This is a *critical* concept.
* **Inference (Generation):** You feed "hello", the model predicts "world". You then *append* "world" to the input and feed "hello world" to predict the *next* token. This is slow (one forward pass per token).
* **Training (Teacher Forcing):** We want to train in parallel. We feed the *entire* correct sequence (e.g., "hello world how are you") into the model at once. But to *prevent cheating*, we must apply a mask. When the model is at position 2 ("how") and trying to *predict* "how", it must only be allowed to *see* "hello" and "world" (positions 0 and 1). If it could see "how", "are", or "you", it would just learn to copy the input, and it would fail at generation time (when it *doesn't* have the future tokens).
* **Summary:** Use Masked Self-Attention when your goal is **generation** or any **auto-regressive** task where the "future" is unknown.

### A Note on Padding Masks

There is a *second* type of mask that is often used in *both* regular and masked self-attention: the **Padding Mask**.

* **Problem:** We train on *batches* of sentences. To make them all the same length in a batch, we "pad" the shorter sentences with a special `<PAD>` token.
* `["hello", "world", "<PAD>", "<PAD>"]`
* `["the", "quick", "brown", "fox"]`
* **Solution:** We don't want the words to "pay attention" to the meaningless `<PAD>` tokens.
* **How:** We create a **padding mask** (e.g., `[1, 1, 0, 0]` for the first sentence) and apply it (just like the causal mask, by setting pad-token *columns* to $-\infty$) before the softmax.
* This ensures that no token, in any position, will assign any attention weight to a `<PAD>` token.

**In a decoder, you often apply *both* masks:**

1.  A **causal mask** (lower-triangular) to prevent looking at the future.
2.  A **padding mask** to prevent looking at padding.
The final mask is the combination of these two.

| Feature | Self-Attention (Encoder) | Masked Self-Attention (Decoder) |
| --- | --- | --- |
| **Primary Goal** | Representation / Understanding | Generation / Auto-regression |
| **Information Flow** | Bi-directional (sees left and right) | Causal / Uni-directional (sees left only) |
| **Attention Matrix** | Dense (all $i, j$ can be non-zero) | Lower-triangular (all $j > i$ are zero) |
| **Mechanism** | Standard Scaled Dot-Product | Dot-Product + "Look-Ahead" Mask |
| **Example Model** | **BERT**, **RoBERTa** (Encoders) | **GPT**, **LLaMA** (Decoders) |
| **Analogy** | Reading a paragraph | Writing a sentence |

</details>

<details>
<summary><strong>5. The Matrix Math for Calculating Masked Self-Attention</strong></summary>

### Building on Self-Attention

The good news is that 90% of the math for Masked Self-Attention is *identical* to standard Self-Attention. We still use the same core formula:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V
$$The *only* difference is the addition of one new component: a **Mask Matrix**.

The modified "unofficial" formula looks like this:

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \textbf{Mask}\right)V
$$Let's walk through the *full* process, highlighting the new step.

We start with our input $X$ (shape $[L, d\_{model}]$, $L$ is sequence length).

**Step 1: Create $Q$, $K$, and $V$**
*This is identical.* We use $W^Q, W^K, W^V$ to project $X$.

* $Q = X * W^Q$ (shape $[L, d_k]$)
* $K = X * W^K$ (shape $[L, d_k]$)
* $V = X * W^V$ (shape $[L, d_v]$)

**Step 2: Calculate Scores ($QK^T$)**
*This is identical.* We perform the batch matrix multiplication.

* `Scores` = $Q * K^T$
* `Scores` shape: $[L, L]$

**Step 3: Scale ($\sqrt{d_k}$)**
*This is identical.* We scale to prevent softmax saturation.

* `ScaledScores` = `Scores` / $\sqrt{d_k}$
* `ScaledScores` shape: $[L, L]$

**Step 4 (NEW): Apply the Look-Ahead Mask**

This is the new, critical step. We must *prevent* the softmax from assigning any probability to "future" tokens. We do this by adding a "mask" matrix to our `ScaledScores`.

* **What is the Mask?**
    It's a matrix, let's call it $M$, with the *same shape* as `ScaledScores`: $[L, L]$. The mask $M$ is defined as:
    * $M[i, j] = 0$ if $j \le i$ (if the key-token $j$ is in the "past" or "present" of query-token $i$)
    * $M[i, j] = -\infty$ if $j > i$ (if the key-token $j$ is in the "future" of query-token $i$)

* **Example for $L=4$:**
    $$
    M = \begin{bmatrix}
    0 & -\infty & -\infty & -\infty \\
    0 & 0 & -\infty & -\infty \\
    0 & 0 & 0 & -\infty \\
    0 & 0 & 0 & 0
    \end{bmatrix}
    $$
    This is an **upper-triangular mask** (excluding the diagonal).

* **The Math:** We add this mask to our `ScaledScores`.
    `MaskedScores` = `ScaledScores` + $M$

* **Why does this work?**
    Let's look at Row 1 (for $i=1$):
    `[score_{1,0}, score_{1,1}, score_{1,2}, score_{1,3}]` + `[0, 0, -inf, -inf]`
    = `[score_{1,0}, score_{1,1}, -inf, -inf]`

    When we feed this vector into `softmax`, we get:
    $\text{softmax}([score_{1,0}, score_{1,1}, -\infty, -\infty])$

    Recall that $softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$.

    The term for $x_2$ will be $e^{-\infty}$, which is $0$.
    The term for $x_3$ will be $e^{-\infty}$, which is $0$.

    The resulting probability distribution will be:
    `[prob_0, prob_1, 0, 0]`

    We have successfully **forced the attention mechanism to assign 0% probability** to words at positions 2 and 3 when it is processing the word at position 1. The information flow to the "future" is cut off.

---

**Step 5: Softmax**

*This is almost identical*, but now it's applied to the *masked* scores.

`AttentionWeights` = $\text{softmax}(\text{MaskedScores})$

* Shape is still $[L, L]$.
* But, the `AttentionWeights` matrix is now **lower-triangular**. All entries above the main diagonal are $0$.
    $$
    \text{AttentionWeights} = \begin{bmatrix}
    w_{0,0} & 0 & 0 & 0 \\
    w_{1,0} & w_{1,1} & 0 & 0 \\
    w_{2,0} & w_{2,1} & w_{2,2} & 0 \\
    w_{3,0} & w_{3,1} & w_{3,2} & w_{3,3}
    \end{bmatrix}
    $$
    (Where $w_{i,0} + w_{i,1} + ... = 1$)

---

**Step 6: Apply to Values ($...V$)**

*This is identical.* We perform the final matrix multiplication.

`Output` = `AttentionWeights` $*$ $V$

* `Output` shape: $[L, d_v]$

---

### The Result

The output $Z$ is (like before) a matrix of contextualized embeddings. However, the context is *causally restricted*.

* $Z_0$ (output for word 0) is a weighted sum of *only* $V_0$.
* $Z_1$ (output for word 1) is a weighted sum of $V_0$ and $V_1$.
* $Z_2$ (output for word 2) is a weighted sum of $V_0$, $V_1$, and $V_2$.
* ...
* $Z_i$ (output for word $i$) is a weighted sum of $V_0, ..., V_i$.

This output $Z$ is now "safe" to be passed to the next layer (e.g., a feed-forward network) or used to predict the *next* token, because we have guaranteed that the representation for word $i$ *only* contains information from words $0...i$, and *not* $i+1...L$. We have preserved the auto-regressive property, even while processing the entire sequence in parallel.

</details>

<details>
<summary><strong>6. Coding Masked Self-Attention in PyTorch</strong></summary>

### Modifying Our `SelfAttention` Class

We don't need a new class. We can make a small, powerful modification to our existing `SelfAttention` class from Topic 3. We will modify the `forward` method to accept an optional `mask` argument.

If a `mask` is provided, we will apply it. This makes our class flexible enough to handle *both* regular self-attention (by passing `mask=None`) and masked self-attention (by passing a mask).

### Creating the Mask in PyTorch

First, how do we create the look-ahead mask $M$ (the one with $0$s and $-\infty$s)?
In PyTorch, we don't typically use $-\infty$. Instead, we use `float('-inf')` or a very large negative number like `-1e9` (which is numerically stable and has the same effect in softmax).

The easiest way to create the mask is to first create a *boolean* mask.

```python
# Let's say seq_len = 5
seq_len = 5

# 1. Create an upper-triangular matrix of 1s (True)
#    torch.triu(..., diagonal=1) gives 1s *above* the diagonal.
#    [[0, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 1],
#     [0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0]]
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

# 2. Convert to boolean
#    True where we want to mask (the 1s)
#    False where we want to keep (the 0s)
bool_mask = (mask == 1)

# --- Output of bool_mask ---
# [[False,  True,  True,  True,  True],
#  [False, False,  True,  True,  True],
#  [False, False, False,  True,  True],
#  [False, False, False, False,  True],
#  [False, False, False, False, False]]
```

This `bool_mask` is perfect. `True` means "this position is in the future, please mask it."

### The `masked_fill_` Operation

PyTorch has a wonderful in-place function: `tensor.masked_fill_(mask, value)`.
It looks at the `mask` tensor. Wherever the `mask` is `True`, it fills the *original* `tensor` with the specified `value`.

`scores.masked_fill_(bool_mask, -1e9)`

This will take our `scores` matrix and, wherever `bool_mask` is `True` (i.e., for all $j > i$), it will replace the score with `-1e9`. This is *exactly* the operation "ScaledScores + M" from the math section.

### The Modified Code

Let's update our `SelfAttention` class. We'll rename it to `Attention` to be more general, and add the `mask` parameter.

```python
import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    """
    A flexible single-head attention module that can handle
    self-attention, masked self-attention, and cross-attention.
    """
    
    def __init__(self, embed_size, d_k=None):
        super(Attention, self).__init__()
        
        self.embed_size = embed_size
        
        if d_k is None:
            d_k = embed_size
            
        self.d_k = d_k
        
        self.query_linear = nn.Linear(embed_size, d_k, bias=False)
        self.key_linear = nn.Linear(embed_size, d_k, bias=False)
        self.value_linear = nn.Linear(embed_size, d_k, bias=False)
        
        self.scale_factor = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        The forward pass of the attention layer.
        
        Args:
            query (torch.Tensor): The Query tensor.
                                  Shape: (batch_size, query_len, embed_size)
            key (torch.Tensor): The Key tensor.
                                Shape: (batch_size, key_len, embed_size)
            value (torch.Tensor): The Value tensor.
                                  Shape: (batch_size, value_len, embed_size)
                                  (Note: key_len == value_len)
            mask (torch.Tensor, optional): A boolean mask.
                                           - For masked self-attention,
                                             shape: (query_len, key_len)
                                           - For padding mask,
                                             shape: (batch_size, 1, key_len)
                                           'True' values will be masked.
                                 
        Returns:
            torch.Tensor: The output tensor (Z).
                          Shape: (batch_size, query_len, d_k)
            torch.Tensor: The attention weights.
                          Shape: (batch_size, query_len, key_len)
        """
        
        # 1. Project to Q, K, V
        # Note: We now use the 'query', 'key', 'value' inputs.
        # For self-attention, query=key=value=x_in
        
        Q = self.query_linear(query) # Shape: (batch, query_len, d_k)
        K = self.key_linear(key)     # Shape: (batch, key_len, d_k)
        V = self.value_linear(value)   # Shape: (batch, value_len, d_k)
        
        # 2. Calculate Scores
        # Q shape: (batch, query_len, d_k)
        # K.transpose(-2, -1) shape: (batch, d_k, key_len)
        # scores shape: (batch, query_len, key_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 3. Scale
        scaled_scores = scores / self.scale_factor
        
        # 4. (MODIFIED) Apply Mask
        if mask is not None:
            # The mask needs to be broadcastable to the scores.
            # scores shape: (batch, query_len, key_len)
            # A (query_len, key_len) mask will be broadcast
            # across the batch dimension.
            # A (batch, 1, key_len) mask (padding mask) will
            # be broadcast across the query_len dimension.
            scaled_scores = scaled_scores.masked_fill(mask, -1e9)
        
        # 5. Softmax
        # attention_weights shape: (batch, query_len, key_len)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # 6. Apply to Values
        # attention_weights shape: (batch, query_len, key_len)
        # V shape: (batch, value_len, d_k)  (and value_len == key_len)
        # output shape: (batch, query_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

### Example Usage (Masked Self-Attention)

Now, let's *use* this module to perform masked self-attention.

```python
# --- Example ---
B, L, E = 2, 10, 512 # Batch, Seq_Len, Embed_size
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Create the input
sample_input = torch.rand(B, L, E).to(device)

# 2. Instantiate the (now flexible) attention layer
# We rename our class to 'Attention'
attention_layer = Attention(embed_size=E, d_k=E).to(device)

# 3. Create the Look-Ahead Mask
# This mask is *not* dependent on the batch, only on sequence length.
seq_len = sample_input.shape[1]
look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

print("--- Look-Ahead Mask (showing 5x5) ---")
print(look_ahead_mask[:5, :5])

# 4. Run the forward pass
# For SELF-attention, query, key, and value are all the same input
# We pass the mask we just created.
output, weights = attention_layer(sample_input, 
                                  sample_input, 
                                  sample_input, 
                                  mask=look_ahead_mask)

print("\n--- Output Shape ---")
print(output.shape)

print("\n--- Attention Weights Shape ---")
print(weights.shape)

print("\n--- Checking the mask (1st sentence, 2nd word) ---")
# The 2nd word (row 1) should *only* attend to words 0 and 1.
# All weights for words 2, 3, 4... should be 0.
print(weights[0, 1, :])
# Expected: [0.52, 0.48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (approx)

print("\n--- Checking the mask (4th word) ---")
# The 4th word (row 3) should only attend to words 0, 1, 2, 3.
print(weights[0, 3, :])
# Expected: [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (approx)

# --- Expected Output ---
# --- Look-Ahead Mask (showing 5x5) ---
# tensor([[False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True],
#         [False, False, False, False, False]], device='cpu')
#
# --- Output Shape ---
# torch.Size([2, 10, 512])
#
# --- Attention Weights Shape ---
# torch.Size([2, 10, 10])
#
# --- Checking the mask (1st sentence, 2nd word) ---
# tensor([5.0911e-01, 4.9089e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#        grad_fn=<SliceBackward0>)
#
# --- Checking the mask (4th word) ---
# tensor([1.7584e-01, 2.7663e-01, 2.3787e-01, 3.0966e-01, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#        grad_fn=<SliceBackward0>)
```

As you can see, the weights matrix is correctly zeroed-out for all $j > i$, confirming our masked self-attention is working perfectly. Our single `Attention` class can now be used in both an Encoder and a Decoder!

</details>

<details>
<summary><strong>7. Encoder-Decoder Attention (Cross-Attention)</strong></summary>

### The Third Type of Attention

We've now seen two of the three attention mechanisms in a standard Transformer:

1.  **Self-Attention:** Used in the **Encoder**. $Q, K, V$ all come from the *same* source (the previous layer's output). It builds a bi-directional understanding of the *input* sequence.
2.  **Masked Self-Attention:** Used in the **Decoder**. $Q, K, V$ all come from the *same* source (the previous *decoder* layer's output). It builds a causal, auto-regressive understanding of the *output* sequence generated so far.

But how does the Decoder *know* what the Encoder was thinking? How does it "read" the input sentence to guide its translation? This is the job of the third mechanism: **Encoder-Decoder Attention**, also known as **Cross-Attention**.

### Where Does it Live?

Cross-Attention lives *inside the Decoder layer*, squished between the Masked Self-Attention and the Feed-Forward Network.

A standard **Decoder Layer** looks like this:

1.  **Masked Self-Attention** (Decoder attends to itself)
2.  Add & Norm
3.  **Cross-Attention** (Decoder attends to Encoder)
4.  Add & Norm
5.  Feed-Forward Network
6.  Add & Norm

### The Core Idea: $Q$ from Decoder, $K$ & $V$ from Encoder

This is the most important concept to grasp. We still use the same $Attention(Q, K, V)$ formula, but the *sources* of $Q$, $K$, and $V$ are different.

  * **Query ($Q$)**: The **Decoder** is asking the questions. The $Q$ matrix comes from the *output* of the previous sub-layer (the Masked Self-Attention).

      * **Analogy:** The "translator" (Decoder), having just written the word "la", forms a new query: "Okay, I've written 'la'. Now, what part of the *source text* ('the house') is most relevant for my *next* word?"
      * $Q = X_{decoder} * W^Q$

  * **Key ($K$) & Value ($V$)**: The **Encoder** provides the answers. The $K$ and $V$ matrices come from the *final output of the entire Encoder stack*. This output is computed *once* and then re-used by *every* Decoder layer at *every* time step.

      * **Analogy:** The "encyclopedia" (Encoder's output) of the source text. "house" has a Key ("I'm a noun, a building") and a Value ("This is my full contextual meaning vector"). "the" has a Key ("I'm a determiner") and a Value ("This is my vector").
      * $K = X_{encoder} * W^K$
      * $V = X_{encoder} * W^V$

**The Process:**
The Decoder's Query $Q$ (from "la") is compared against *all* the Encoder's Keys $K$ (from "the" and "house").

  * $Q_{"la"}$ vs $K_{"the"}$ -\> High score
  * $Q_{"la"}$ vs $K_{"house"}$ -\> Low score
    The softmax is computed, and the new vector is a weighted sum of the Encoder's *Values*.
  * `Output = 0.9 * V_{"the"} + 0.1 * V_{"house"}`
    This output vector, which has "absorbed" the relevant information from the encoder, is then passed to the decoder's Feed-Forward network.

### The Matrix Math (Shapes)

This is where it gets interesting, as the sequence lengths can be different\!

  * Let $L_{in}$ be the input sequence length (e.g., "the house", $L_{in}=2$).
  * Let $L_{out}$ be the output sequence length (e.g., "la maison", $L_{out}=2$).

<!-- end list -->

1.  **Encoder Output**: $X\_{encoder}$ has shape `(batch, L_in, d_model)`.
2.  **Decoder Input**: $X\_{decoder}$ (from masked self-attention) has shape `(batch, L_out, d_model)`.

Now, let's feed them into our `Attention` module (from the previous topic) using `forward(query, key, value)`:

  * `query` = $X_{decoder}$ (shape `[B, L_out, d_model]`)
  * `key` = $X_{encoder}$ (shape `[B, L_in, d_model]`)
  * `value` = $X_{encoder}$ (shape `[B, L_in, d_model]`)

Let's trace the `forward` pass:

1.  `Q = self.query_linear(query)` -\> Shape: `[B, L_out, d_k]`

2.  `K = self.key_linear(key)` -\> Shape: `[B, L_in, d_k]`

3.  `V = self.value_linear(value)` -\> Shape: `[B, L_in, d_v]` (where $d\_v=d\_k$)

4.  `scores = torch.matmul(Q, K.transpose(-2, -1))`

      * `Q` shape: `[B, L_out, d_k]`
      * `K.transpose` shape: `[B, d_k, L_in]`
      * `scores` shape: `[B, L_out, L_in]`

This `scores` matrix (the attention weights) is the key. Its shape is `[L_out, L_in]`.

  * `scores[i, j]` represents: "For the $i$-th **output** word, how much attention should we pay to the $j$-th **input** word?"
  * This is the "alignment" that the original RNN-with-attention models sought\! It's learned completely automatically.

<!-- end list -->

5.  `attention_weights = torch.softmax(scores, dim=-1)` -\> Shape: `[B, L_out, L_in]`

6.  `output = torch.matmul(attention_weights, V)`

      * `attention_weights` shape: `[B, L_out, L_in]`
      * `V` shape: `[B, L_in, d_v]`
      * `output` shape: `[B, L_out, d_v]`

The output has the *decoder's* sequence length (`L_out`) but is now infused with information from the *encoder* (the weighted sum of $V\_{encoder}$). This `output` is what gets passed to the decoder's feed-forward layer.

**Masking in Cross-Attention:**

  * **Do we use a look-ahead mask?** **NO.** The decoder should be allowed to look at the *entire* input sentence at every step.
  * **Do we use a padding mask?** **YES.** If the *input* sentence was padded (e.g., `["the", "house", "<PAD>", "<PAD>"]`), we must pass a padding mask for the *encoder's* $K$ and $V$. This mask would tell the decoder to *ignore* the `<PAD>` tokens from the encoder, so they are never assigned any attention.

</details>

<details>
<summary><strong>8. Multi-Head Attention</strong></summary>

### The Problem with Single-Head Attention

So far, we have built a powerful `Attention` module. But it has one potential weakness.

Our module learns *one* set of $W^Q, W^K, W^V$ matrices. This means it learns *one* way to "pay attention." It might learn to focus on subject-verb agreement, for example.

But what if a sentence requires *multiple* types of relationships to be understood simultaneously?

  * Subject-Verb agreement ("The **cats**... **are**...")
  * Adjective-Noun modification ("The **quick brown**... **fox**...")
  * Prepositional phrases ("...jumped **over** the **lazy dog**")

A single attention "head" might struggle to learn all these different patterns at once. It's forced to average its attention.

### The Solution: Multi-Head Attention

The solution proposed in "Attention is All You Need" is simple and elegant: **run *multiple* attention "heads" in parallel, and then combine their results.**

  * **Analogy:** Instead of having one expert (a single head) read a sentence and give you a summary, you assemble a *committee* of 8 different experts ($h=8$ heads).
      * Head 1 (Linguist): Focuses on grammatical structure.
      * Head 2 (Historian): Focuses on the historical context of words.
      * Head 3 (Poet): Focuses on metaphorical relationships.
      * ...
  * Each expert reads the *same* sentence. Each produces their *own* small summary (their output $Z_i$).
  * At the end, you just *concatenate* all 8 small summaries together to form one large, comprehensive summary.
  * Finally, you pass this concatenated summary to a "manager" (a final linear layer, $W^O$) who synthesizes them into a single, coherent final output.

### The Matrix Math (Conceptual)

Let's say we have $h=8$ heads and $d_{model}=512$.

1.  We don't just have *one* $W^Q, W^K, W^V$. We now have **8** sets of them:

      * $(W^Q_1, W^K_1, W^V_1)$ for $Head_1$
      * $(W^Q_2, W^K_2, W^V_2)$ for $Head_2$
      * ...
      * $(W^Q_8, W^K_8, W^V_8)$ for $Head_8$

2.  We "split" the $d_{model}$ across the heads. The dimension for *each head* is $d_k = d_v = d_{model} / h$.

      * $d_k = 512 / 8 = 64$.
      * So, each $W^Q_i$ matrix projects from $d_{model}=512$ down to $d_k=64$.

3.  We feed the *same* input $X$ to *all 8 heads* in parallel.

    * $Q_i = X * W^Q_i$ (shape `[L, 64]`)
    * $K_i = X * W^K_i$ (shape `[L, 64]`)
    * $V_i = X * W^V_i$ (shape `[L, 64]`)

4.  We run the attention formula for *each head* independently.

      * $Head_i = Attention(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{64}}\right)V_i$
      * The output of *each head* ($Head_i$) is a matrix of shape `[L, 64]`.

5.  **Concatenate:** We stick the 8 output matrices together side-by-side.

      * $Concat = \text{Concat}(Head_1, Head_2, ..., Head_8)$
      * The shape of $Concat$ is `[L, 8 * 64]` = `[L, 512]`. We are back to our original $d\_{model}$\!

6.  **Final Linear Layer:** We apply one final, learnable matrix $W^O$ (shape `[512, 512]`) to the concatenated output. This allows the model to *mix* the information learned by the different heads.

      * $Output = Concat * W^O$
      * Final $Output$ shape: `[L, 512]`.

### The Matrix Math (Implementation)

In practice, we don't *actually* create 8 separate linear layers and run 8 loops. That would be slow. We can do this *all at once* with clever matrix algebra.

1.  Instead of 8 $W^Q_i$ matrices of shape `[512, 64]`, we create *one* big $W^Q$ matrix of shape `[512, 512]` (which is just the 8 small ones stacked).

2.  `Q = X * W^Q` -\> $Q$ has shape `[Batch, L, 512]`

3.  `K = X * W^K` -\> $K$ has shape `[Batch, L, 512]`

4.  `V = X * W^V` -\> $V$ has shape `[Batch, L, 512]`

5.  **This is the magic trick:** We now *reshape* and *transpose* these $Q, K, V$ matrices to "reveal" the heads.

      * `Q = Q.reshape(Batch, L, 8_heads, 64_head_dim)`
      * `Q = Q.transpose(1, 2)`
      * The new shape of $Q$ is `[Batch, 8_heads, L, 64_head_dim]`

    We do the same for $K$ and $V$. Why this shape? Because now, `torch.matmul` will treat the `Batch` and `8_heads` dimensions as "batch" dimensions. It will *simultaneously* perform 8 separate matrix multiplications of shape `[L, 64]` and `[64, L]`\!

6.  **Run Attention (Batched):**

      * `scores = torch.matmul(Q, K.transpose(-2, -1))`
      * `Q` shape: `[B, h, L, dim]`
      * `K.T` shape: `[B, h, dim, L]`
      * `scores` shape: `[B, h, L, L]`
      * We now have 8 attention maps, one for each head\!

7.  `scaled_scores = scores / math.sqrt(64)`

8.  (Apply mask here, if any. The mask must be broadcast to `[B, h, L, L]`)

9.  `attention_weights = torch.softmax(scaled_scores, dim=-1)` -\> Shape `[B, h, L, L]`

10. `output = torch.matmul(attention_weights, V)`

      * `weights` shape: `[B, h, L, L]`
      * `V` shape: `[B, h, L, dim]`
      * `output` (our $Head_{i}\text{s}$) shape: `[B, h, L, dim]`

11. **Reverse the magic trick:** We "un-transpose" and reshape to concatenate.

      * `output = output.transpose(1, 2)` -\> Shape `[B, L, h, dim]`
      * `output = output.contiguous().reshape(B, L, h * dim)`
      * `output` shape: `[B, L, 512]` (This is our $Concat$ matrix)

12. **Final Linear Layer:**

      * We have a `self.out_linear = nn.Linear(512, 512)` (our $W^O$).
      * `final_output = self.out_linear(output)` -\> Shape `[B, L, 512]`.

This is Multi-Head Attention. It's computationally similar to single-head attention (since $8 \times [512, 64]$ is roughly the same number of parameters as $[512, 512]$), but it's *representationally* much more powerful. This is the standard building block of all Transformers.

</details>

<details>
<summary><strong>9. Coding Encoder-Decoder Attention and Multi-Head Attention in PyTorch</strong></summary>

### The Final Module: `MultiHeadAttention`

We are now ready to combine all the concepts:

1.  **Multi-Head:** Splitting $d_{model}$ into $h$ heads.
2.  **Flexible $Q, K, V$:** Allowing for Self-Attention ($Q=K=V$) and Cross-Attention ($Q \neq K, V$).
3.  **Masking:** An optional `mask` argument to handle both Causal and Padding masks.

This *one module* will be the workhorse we can use for all three attention-based sub-layers in the Transformer.

### The Full, Annotated Code

Here is the complete `MultiHeadAttention` class, implementing the efficient, batched approach described in the previous topic.

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention as described in
    "Attention is All You Need".
    
    This module is flexible and can be used for:
    1. Self-Attention (in Encoder): 
       forward(x, x, x, mask=padding_mask)
    2. Masked Self-Attention (in Decoder):
       forward(x, x, x, mask=combined_mask)
    3. Cross-Attention (in Decoder):
       forward(x_decoder, x_encoder, x_encoder, mask=padding_mask)
    """
    
    def __init__(self, embed_size, num_heads, dropout_rate=0.1):
        """
        Args:
            embed_size (int): The d_model, total dimensionality of the
                              input/output embeddings.
            num_heads (int): The number of attention heads (h).
            dropout_rate (float): Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_size % num_heads == 0, \
            "Embedding size (d_model) must be divisible by num_heads"
            
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # d_k
        
        # We create the "big" W^Q, W^K, W^V matrices
        # In PyTorch, nn.Linear(in, out) is x*W^T + b.
        # We will create one big linear layer for each
        self.query_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.key_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.value_linear = nn.Linear(embed_size, embed_size, bias=False)
        
        # The final output linear layer (W^O)
        self.out_linear = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.scale_factor = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            query (torch.Tensor): Query tensor.
                                  Shape: (batch_size, query_len, embed_size)
            key (torch.Tensor): Key tensor.
                                Shape: (batch_size, key_len, embed_size)
            value (torch.Tensor): Value tensor.
                                  Shape: (batch_size, value_len, embed_size)
            mask (torch.Tensor, optional): Boolean mask. 
                                           True indicates a position to be masked.
                                           Shape: (batch, 1, 1, key_len) for padding,
                                                  (1, 1, query_len, key_len) for causal,
                                                  or broadcastable.
                                 
        Returns:
            torch.Tensor: The output tensor.
                          Shape: (batch_size, query_len, embed_size)
        """
        
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1] # Should be == key_len
        
        # 1. Pass inputs through linear layers
        # Q, K, V shape: (batch_size, seq_len, embed_size)
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)
        
        # 2. Reshape and Transpose to split into heads
        # We reshape from (B, L, d_model) to (B, L, h, d_k)
        # Then transpose to (B, h, L, d_k)
        
        # Q shape: (batch, num_heads, query_len, head_dim)
        Q = Q.reshape(batch_size, query_len, 
                      self.num_heads, self.head_dim).transpose(1, 2)
                      
        # K shape: (batch, num_heads, key_len, head_dim)
        K = K.reshape(batch_size, key_len, 
                      self.num_heads, self.head_dim).transpose(1, 2)
                      
        # V shape: (batch, num_heads, value_len, head_dim)
        V = V.reshape(batch_size, value_len, 
                      self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Calculate Scores (Scaled Dot-Product)
        # Q shape: (B, h, L_q, d_k)
        # K.transpose(-2, -1) shape: (B, h, d_k, L_k)
        # scores shape: (B, h, L_q, L_k)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        
        # 4. Apply Mask (if provided)
        if mask is not None:
            # The mask must be broadcast to the scores shape
            # (B, h, L_q, L_k).
            # A common padding mask shape is (B, 1, 1, L_k)
            # A common causal mask shape is (1, 1, L_q, L_k)
            scores = scores.masked_fill(mask, -1e9)
            
        # 5. Softmax
        # attention_weights shape: (B, h, L_q, L_k)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # (Optional) Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)
        
        # 6. Apply to Values
        # attention_weights shape: (B, h, L_q, L_k)
        # V shape: (B, h, L_v, d_k) (where L_v == L_k)
        # output shape: (B, h, L_q, d_k)
        output = torch.matmul(attention_weights, V)
        
        # 7. Reverse the Reshape/Transpose
        # We need to get back to (B, L_q, d_model)
        
        # output shape: (B, L_q, h, d_k)
        output = output.transpose(1, 2).contiguous()
        
        # .contiguous() is needed after transpose to use .reshape()
        # output shape: (B, L_q, embed_size)
        output = output.reshape(batch_size, query_len, self.embed_size)
        
        # 8. Final Linear Layer (W^O)
        # output shape: (B, L_q, embed_size)
        final_output = self.out_linear(output)
        
        return final_output
```

## Dissecting the Code

* **`__init__`**:
    * `assert embed_size % num_heads == 0`: This is a critical sanity check. If our $d_{model}=512$ is not divisible by $h=8$, the math for `head_dim` breaks.
    * `self.head_dim = embed_size // num_heads`: This is $d_k$.
    * `nn.Linear(embed_size, embed_size)`: We define the *three* big projection matrices $W^Q, W^K, W^V$ and the *one* final output matrix $W^O$.

* **`forward`**:
    * The signature `(self, query, key, value, mask=None)` is the key to its flexibility.
    * `Q = Q.reshape(...).transpose(1, 2)`: This is the "implementation trick" from Topic 8. We reshape to `(B, L, h, d_k)` and then transpose to `(B, h, L, d_k)`. This makes the `h` dimension a "batch" dimension, so all 8 heads are computed in parallel.
    * `scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor`: This is the batched, scaled dot-product.
    * `scores = scores.masked_fill(mask, -1e9)`: This is the *most* important part for masking. PyTorch's broadcasting rules will handle the mask.
        * **Causal Mask:** We create a mask of shape `(1, 1, L_q, L_k)`. PyTorch broadcasts this to `(B, h, L_q, L_k)`, applying the *same* causal mask to all heads and all batches.
        * **Padding Mask:** We create a mask of shape `(B, 1, 1, L_k)`. PyTorch broadcasts this to `(B, h, L_q, L_k)`, applying the *correct* padding mask for each item in the batch to all heads.
    * `output = output.transpose(1, 2).contiguous()`: We swap the `L_q` and `h` dimensions back. We call `.contiguous()` because `transpose` can create a non-contiguous view of the tensor, which `reshape` cannot handle. `.contiguous()` creates a new, contiguous tensor in memory.
    * `output = output.reshape(...)`: This is the "concatenation" step. We flatten the `h` and `d_k` dimensions back into the single `embed_size` dimension.
    * `final_output = self.out_linear(output)`: We apply the final $W^O$ matrix to mix the head-information.

With this single `MultiHeadAttention` class, we have implemented the core "heart" of the Transformer. We can now stack these modules, along with Feed-Forward layers and Layer-Normalization, to build a full Encoder and Decoder.

</details>

## Acknowledgement

This repository is for personal learning and educational purposes only, serving as a supplement to the "Attention in Transformers: Concepts and Code in PyTorch" course.

All course materials, content, and licenses are the property of **DeepLearning.AI** and **StatQuest**. This repository does not claim any ownership of the original course content.