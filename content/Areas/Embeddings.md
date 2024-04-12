### Introduction
An embedding is a map from $\mathbb{R}^d \to \mathbb{R}^e$ where $d$ is the dimension of a [[One-Hot Encoding]] of a model input and $e$ is the dimension of the latent space. 

The most useful aspect of embeddings is that they retain semantic meaning. That is, if two vectors are close to each other, typically via [[Cosine Similarity]], then these two vectors are similar as objects within the dataset. This is in completely contrast to a one-hot encoding where all of the samples within a dataset are orthogonal!

To give an example, we might expect that the embeddings for the words "toaster", "bread", "apple", "truck" have the below relationship.

**Embedding example**
![[embedding_example.svg|400]]

An embedding's ability to retain semantic meaning compliments a neural network's output nicely. To exemplify this, suppose we have a corpus of text and we've numbered each of the unique words that arise in the corpus. We train a neural network to predict the integer numbering of the next word in a sequence of words. To do this, we feed as input a sequence of words of length $n$ in the form of a vector $x = (x_1,\dots, x_n)$ where $x_i\in \mathbb{Z}$ is the integer numbering for the $i$-th word in the sentence. The output of the neural network is the prediction of the next word in the sequence. 

Let's say the output of the neural network is 2082.49 while the actual word should be 2083. The predicted word 2082 and 2083, while close in the numbering, could be substantially different in their meaning. 

### How do we get such an embedding?
To start, assume that we already have all such embeddings for our one-hot encodings. To rapidly map from the one-hot encoding to the embedding, we place them in an <mark style="background: #BBFABBA6;">embedding matrix</mark>. 

**Illustration of an embedding matrix**
![[embedding_matrix.svg|520]]

Naturally, we can now change the question slightly and ask: how do we find this embedding matrix? As with most things in deep learning, we learn the entries! We first need a training set, however. 
### Developing a training set
Given some corpus of text, we must create a training set containing sequences of words. Below are two such methods.
#### Continuous Bags of Words (CBOW)
In the CBOW method, we predict the next word with a sliding window of size $k$. Suppose we have a sequence (a sentence) of length $n$ with words $w_i$ for $i\in [1,n]$. 
$$
(w_1, w_2, w_3, ..., w_n)
$$
Some of the training samples in the dataset are then below.

| Input 1 | Input 2 | ... | Input k   | Output    |
| ------- | ------- | --- | --------- | --------- |
| $w_1$   | $w_2$   |     | $w_k$     | $w_{k+1}$ |
| $w_2$   | $w_3$   |     | $w_{k+1}$ | $w_{k+2}$ |
| $w_3$   | $w_4$   |     | $w_{k+2}$ | $w_{k+3}$ |

Note that for those sequences of words where we do not have enough words to fill the entire window, we can either pad appropriately or omit them. 
#### Skip-grams
The key idea with skip-grams is that instead of only looking forward, we now look backwards as well. Given some window size k, we construct the dataset by sliding a window of length k+1 across the sequence. The individual training samples consist of an input of the center word in the window and an output of one of the other words in the window. 

Consider the sentence

`"The man went to the store to find ice cream"`

and a window size of 4. Then, some of the samples in the dataset would be in the table below.

| Input | Target |
| :---: | :----: |
| went  |  the   |
| went  |  man   |
| went  |   to   |
| went  |  the   |
|  to   |  man   |
|  to   |  went  |
where these correspond to the windows

![[embedding_skip-gram.svg|500]]

### Skip-gram model architecture 
We will proceed using the skip-gram approach, as it is more commonly used (verify this?). Once we create our dataset, we swap all of the words with their respective one-hot encodings and use the below neural network architecture. 

#### Approach #1: Center predicting context word

![[neural_network_skipgram.svg|600]]

For a given input $w_c$, let $\{w_o\}$ be the set of all words outside of the center, i.e. within the context window. We can approximate the probability $P(\{w_o\} \,|\, w_c)$ as 

$P(\{w_o\} \,|\, w_c) = \prod_{i\in\mathrm{window}} P(w_{o_i}\,|\,w_c)$

Assuming a sequence length of $N$ and a window size of $k$, the [[Likelihood Function]] is 
$$
\prod_{n=1}^N\prod_{-k\leq j\leq k, \,j\neq 0} P(w_{n+j}\,|\,w_n).
$$
Naturally, we would like to maximize this likelihood ([[Maximum Likelihood Estimation]]) and so we will minimize the [[Negative Log Likelihood]]. This will then be our loss function, 
$$
\mathcal{L} = -\sum_{n=1}^N\sum_{-k\leq j\leq k, \,j\neq 0} \log P(w_{n+j}\,|\,w_t). 
$$
Let $v_c$ be the row in the embedding matrix for $w_c$ and $u_o$ be the row in the context matrix for the target (output) word in the above model architecture. Then 
$$
P(w_o\,|\,w_c) = \frac{\exp(u_o^Tv_c)}{\sum_{i\in V} \exp(u_i^Tv_c)}
$$
#### Approach #2: Two word inputs predicting closeness
The key idea between approach #1 and #2 is that we are changing from "who is my neighbor?" to "are we neighbors?"

We now will choose $P(D = 1\,|\,w_c,w_o) = \sigma(u_o^Tv_c)$ and will maximize the likelihood
$$
\prod_{n=1}^N\prod_{-k\leq j\leq k, \,j\neq 0} P(D = 1\,|\,w_c,w_o).
$$
Note that with the above approach a trivial classifier, which outputs 1 for everything, will give the best score. This is because all of the targets are 1! For example, a sample from approach #1 such as 
$$
(\mathrm{input}, \mathrm{target}) = (\text{a}, \text{chased})
$$ is changed to 
$$
(\mathrm{input}_1, \mathrm{input}_2, \mathrm{target}) = (\text{a}, \text{chased}, 1).
$$ To fix this, we introduce <mark style="background: #BBFABBA6;">negative sampling</mark>. 
##### Negative Sampling 
For every training sample, $(\mathrm{input}_1, \mathrm{input}_2, 1)$, we append to our dataset samples 
$$
(\text{input}_1, \text{negative}, 0)
$$
where `negative` is some word that does not appear with $\text{input}_1$ anywhere in the dataset. 
### Comparison between CBOW and skip-gram
The end result between CBOW and skip-grams are similar, but differ slightly. CBOW learns better syntactic relationships whereas skip-gram learns better semantic relationships. 

For example, consider the word `cat`. A CBOW embedding of `cat` will most likely be close to embeddings of words that are semantically close such as `cats`. However, a skip-gram embedding of `cat` may be closer to an embedding of `dog`. 

> [!info] 
> For more information on the differences, read the below StackExchange post. 
> https://ai.stackexchange.com/questions/18634/what-are-the-main-differences-between-skip-gram-and-continuous-bag-of-words


