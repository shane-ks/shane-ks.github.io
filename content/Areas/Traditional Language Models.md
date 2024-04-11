### Introduction
In order to process text, we need to break it up into smaller units called <mark style="background: #BBFABBA6;">tokens</mark>. Specifically, a sentence is transformed into a sequence of tokens. All such unique tokens from a particular dataset is the <mark style="background: #BBFABBA6;">vocabulary</mark>. 

**An example of tokenization**
![[language_model_tokens.svg|600]]

We can process these tokens further by also use either <mark style="background: #BBFABBA6;">stemming</mark> or <mark style="background: #BBFABBA6;">lemmatization</mark>. The goal of both is to reduce inflectional forms.

>[!info] Inflectional forms
><mark style="background: #BBFABBA6;">Inflectional</mark>: relating to or involving a change in the form of a word to express a grammatical function or attribute
>
>Some examples of inflectional forms being reverted to their base-case are:
>
>am, are, is $\Longrightarrow$ be
>run, running, runs $\Longrightarrow$ run

### Stemming
Stemming is the process of reducing words to their roots, usually by removing prefixes and suffixes.

>[!example] Stemming example
> (a) car, cars, car's, cars', care $\Longrightarrow$ car 
> (b) jumping, jumps, jumped $\Longrightarrow$ jump
> (c) likely, likes, liked $\Longrightarrow$ like
> (d) fish, fisher, fishing, fished $\Longrightarrow$ fish
> (e) argue, argues, argued, arguing $\Longrightarrow$ argu

The downside with stemming is that it is not context-aware, as in (a). Moreover, the stem need not be a word at all, as in (e). 

### Lemmatization
Lemmatization is similar to stemming, except it is context-aware and preserves the semantic meaning of the word. 

>[!example] Lemmatization example
>car, car, car's, cars' $\Longrightarrow$ car
>care, caring, cared $\Longrightarrow$ care

Since it is more complex than stemming, it is slower as well. 
### Tokenization
Tokenization is the process of creating tokens from text. There are several approaches that we could take to tokenize our text. 

>[!info] Info
>By this stage, it is common that punctuations are removed and all characters are lowercase.
#### 1. Whitespace Tokenization
We split the words on whitespaces only. Note that in this approach, hyphenated phrases and conjunctions are not split. 

#### 2. Sub-word Tokenization
We split the words on statistically significant fragments. That is, we find the fragments that appear most frequently in the text, and split on that. The downside in this approach is that the tokens lose their direct interpretability. 

Sub-word tokenization is actually the *de facto* standard for tokenization in neural language models. There are several upsides to using sub-word tokenization:
- shorter encodings of frequent tokens;
- composability of subwords;
- ability to deal with unknown words.

These upsides come from the fact that we are interpolating between word-based and character-based tokenization. Common words will appear in the vocabulary. But, if we encounter unknown words, then we can break this word into word fragments or even individual characters. 

>[!example] Sub-word tokenization example
>"I have a new GPU!" $\Longrightarrow$ ["i", "have", "a", "new", "gp", "u", "!"]
>
>The "u" was split from "gpu" since "gpu" is not contained within the vocabulary, but "gp" and "u" are.

### Language Modeling
A language model estimates the probability of any sequence of words. First, it is valuable to examine how we can model sequential data in general. Let $p(x_1,\dots, x_n)$ be the joint distribution of the measurements $x_i$ for $i\in[1,n]$. Then, we can model sequential data as 
$$
\tag{A}
P(x_1, \dots, x_T) = \prod_{t=1}^T P(x_t\,|\,x_{t-1},\dots,x_1)
$$
The above equality is just an application of the [[Law of Total Probability]]. We can then naturally expand this to modeling a sequence of words by just taking $x_i$ to be the $i$-th token in a sequence of words. 
#### Unigrams 
The naive approach to modeling language is to assume that each word is independent. That is, the product of conditional probabilities collapses into the product 
$$
\tag{B}
P(x_1, \dots, x_T) = \prod_{t=1}^T P(x_t)
$$
of marginal probabilities due to independence. 

Some slight care must be take in calculating the marginal probability $P(x_i)$ due to the case where $x_i$ does not appear in the corpus $W$. For instance, if we were to calculate $P(x_i)$ as $\displaystyle P(x_i) = \frac{n_W(x_i)}{|W|}$, which is the intuitive way to calculate this probability, then $P(x_i) = 0$ exactly when $x_i$ is not contained in $W$. This forces the entire product in (B) to be 0. 

Instead, we calculate the marginal probabilities as 
$$
\tag{C}
P(x_i) = \frac{n_W(x_i) + \alpha}{|W| + \alpha|V|}
$$
where $|V|$ is the cardinality of the set containing all unique words in the training corpus $W$ and $\alpha$ is a smoothing constant. Typically, $\alpha$ is kept small and between $0.2$ and $0.5$. This is technique is called <mark style="background: #BBFABBA6;">additive smoothing</mark>. Now, if $x_i\notin W$, then we replace $x_i$ with "UNK" and get the below equation. 
$$
P(x_i) = P(\mathrm{UNK}) = \frac{\alpha}{|W| + \alpha|V|} > 0
$$
##### Downsides of the Unigram Approach
###### 1. Context is not used.
The sentences 
- "I ran to catch the bus."
- "The bus ran to catch I."
are equally likely under the unigram approach. 
###### 2. The next predicted word in any sequence is the most common word in the corpus.
The most common word in the english language is "the." So, most likely, the next predicted word will always be "the," regardless of the sentence.

#### Bigrams
A natural extension of the unigram model is the bigram model, which looks at pairs of words. We use equation (A) but approximate $P(x_t|x_{t-1} , \dots, x_{1})$ with $P(x_t\,|\,x_{t-1})$ and use [[Bayes' Theorem]], 
$$
P(A\,|\,B) = \frac{P(A\cap B)}{P(B)},
$$
to accomplish this. Let's look at an example. Suppose we have the sentence, or sequence of tokens,
$$
X = (x_{1},x_{2},x_{3},x_{4},x_5)
$$
We want to calculate the probability $P(X)$, which using (A) turns into 
$$
P(X) = P(x_1\,|\,\mathrm{<S>})\,P(x_2\,|\,x_1)\,P(x_3\,|\,x_2)\,P(x_4\,|\,x_3)\,P(x_5\,|\,x_4)\,P(\mathrm{</S>}\,|x_5)
$$
where $\mathrm{<S>}$ and $</S>$ are special start and end tokens, respectively. We calculate $P(x_1\,|\,\mathrm{<S>})$ by counting the occurrences where $x_1$ is at the beginning of a sentence divided by the number of total sentences. The same logic can be applied to $P(\mathrm{</S>}\,|x_5)$. For $P(x_{i}\,|\,x_{i-1})$ where $x_i$ and $x_{i-1}$ are non-special tokens within a sentence, we calculate the conditional probability as 
$$
P(x_{i}\,|\, x_{i-1}) = \frac{n_W(x_{i-1}x_i)}{n_W(x_{i-1})}
$$
where $n_W(x_{i-1}x_i)$ is the number of occurrences of $x_{i-1}$ followed by $x_i$ contained in the corpus $W.$ 
#### N-grams Language Model
Generalizing bigrams further, we arrive at $N$-grams. The process is the same as bigrams, except we now condition on $N-1$ tokens. For instance, we have
$$
P(x_1, \dots, x_T) \approx \prod_{t=1}^T P(x_t\,|\,x_{t-1},\dots,x_{t-N+1})
$$
At the beginning and ends of the sentence, we now pad with either $<S>$ or $</S>$ as necessary until we are conditioning always on $N-1$ tokens. 

>[!info] Info
>For more information on $N$-grams, check out:
>https://web.stanford.edu/~jurafsky/slp3/3.pdf

##### Downsides of the N-gram approach
###### 1. Storage issues as $N$ increases
As the window size $N$ increases, the amount of storage necessary explodes. The reason is that we must pre-compute probabilities, which requires we count possible permutation (with replacement) of words in our vocabulary up to the specified window size. 
###### 2. Lack of deep semantic relationships 
By just counting the occurrences of words in relation to others, we do not necessarily capture any semantic difference. For example, we may not know from the counts alone that "vehicle," "car," and "automobile" are all similar. 

To address these shortcomings, we turn to [[Neural Network Language Models]] via [[Embeddings]], which do in fact capture this semantic meaning and overcomes the storage issue.