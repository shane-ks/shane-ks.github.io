### Introduction 
A recurrent neural network (RNN) is a model architecture designed to handle sequential data. They introduce a concept of memory and carry forward the context history. 

An RNN is useful for a variety of tasks, some of which are below.
- Sentiment analysis
- Translation 
- Answering questions
- Text generation

### Model architecture
The key idea with an RNN is that we also feed state into the neural network alongside the input. 

**RNN model diagram**
![[rnn_diagram.svg|500]]

More mathematically, an RNN is governed by the recurrence relation 
$$
h_t = f_{u,v,\beta}(h_{t-1}, x_t, \beta) = tanh(Uh_{t-1} + VX_t + \beta_1),
$$
where $h_0$ is typically all zeros or randomized. We typically will ignore the bias $\beta$ and just write $h_t = f_{u,v}(h_{t-1}, x_t)$. Note that we can technically use any linear and bounded activation function, though we use $tanh$ here. The output $y_t$ is 
$$
Y_t = f(Wh_t + \beta_2),
$$
where $f$ is the activation function dependent on the task. 

**RNN model broken down more**
![[rnn_diagram_pic.png|620]]
In the above diagram, note that there is no bias term for the $Vx_t$ term. This is because we are adding $Vx_t$ to $Uh_{t-1} + \beta_1$, and so any additional bias can be collapsed into $\beta_1$.  

### RNN training 
Suppose we have an input $(x_1, x_2, \dots, x_n)$ along with labels $y_1, \dots, y_n$. To train the RNN, we follow the steps below. 
#### 1. Forward pass
We start by initializing $h_0$. We then calculate $\hat{y}_i$ by feeding in the input $x_i$ and the prior history $h_{i-1}$ for $i\in[1,n]$. 
#### 2. Calculate the loss for each point and aggregate
Given a loss function $L(y, \hat{y})$, we calculate the loss $L_i$ of each $y_i$ and $\hat{y}_i$ for $i\in[1,n]$. We then set 
$$
L = \sum_{t=1}^n L_t
$$
#### 3. Calculate gradients
Note that we have a peculiar situation when it's time to calculate the gradients, namely that there are strings if dependencies from the histories $h_t$. To handle this, we use backpropagation through time. 

**Backpropagation through time**
![[rnn_backprop_time.png]]
We will rigorously derive the partial derivatives in [[Backpropagation Through Time]], but they are 

**Partial derivatives using backpropagation through time**
![[backprop_time_partials.png|500]]

### Issues with RNNs
Due to backpropagation through time, the gradient is a product of many different partials, which causes the gradient to suffer from both the [[Vanishing Gradient Problem]] and the [[Exploding Gradient Problem]]. 

In order to address these problems, we can use [[Long Short-Term Memory (LSTM)]] and [[Gated Recurrent Unit (GRU)]]. 
### Types of RNNs
#### Simple RNN
![[rnn_diagram.svg|500]]
We can implement a simple RNN quite easily in Keras. There is a `SimpleRNN` layer, which we can use as below.

```python
model = Sequential(name='SimpleRNN')
model.add(Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN, mask_zero=True))
model.add(SimpleRNN(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=128)
```
#### Many-to-many (self-supervised)
![[rnn_many_to_many_self_supervised.svg|650]]
#### One-to-many (Supervised)
We can use a one-to-many model architecture to generate a melody from a starting note. 
![[rnn_one_to_many.svg|600]]
#### Many-to-One (Supervised)
![[rnn_many_to_one.png|580]]
#### Many-to-Many (Supervised)
One use case for a many-to-many model architecture is to tag parts of speech. 
![[rnn_many_to_many.png|580]]

### Bidirectional RNNs
The main idea is that sometimes in order to correctly predict, we need to know context of before and after. 

**Diagram of a bidirectional RNN**
![[bidirectional_rnn_diagram.png]]

TODO: CONTINUE
