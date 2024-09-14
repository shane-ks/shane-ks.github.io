### Introduction
Using a neural network for a language model solves some of the issues that [[Traditional Language Models]] had. 
##### Neural network language model upsides
###### 1. No sparsity issues
###### 2. No storage issues

##### Neural network language model downsides
###### 1. Fixed-window size can never be enough
###### 2. The weights awkwardly handle word position 
###### 3. No concept of time. 

### Model architecture
Given a corpus of text, we generate [[Embeddings]] for each word in the vocabulary. We then generate a dataset where the input is a sequence of words and the label is the next word. We follow the model architecture below. 

**Diagram of model architecture**
![[neural_network_language_model_architecture.png]]



### Evaluation 
Perplexity 
