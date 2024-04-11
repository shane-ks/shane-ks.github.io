### Introduction
An autoencoder is a [[Self-Supervised Learning]] system that learns to compress and reconstruct input data. It is split into two parts, an <mark style="background: #BBFABBA6;">encoder</mark> and a <mark style="background: #BBFABBA6;">decoder</mark>.

**Autoencoder Architecture**
![[autoencoder_diagram.svg|600]]

The encoder takes the input and reduces the dimensionality to the <mark style="background: #BBFABBA6;">latent variables</mark>, or the <mark style="background: #BBFABBA6;">encoding</mark>. This encoding is the input to the decoder, which blows it back up to the original input dimensions. The loss is then calculated by comparing the similarity between the input and output.

In other words, the network is trained with the explicit goal of recreating the original input with the important restriction that it flows through the bottleneck. Once trained, we typically throw away the decoder and use the encoder as a specialized means of compression. Alternatively, we can use the encoding and decoder to clean noisy data. 

____
### Autoencoder use-cases

#### 1. Compression / Dimension-reduction
We have already discussed this above, as it is a very common reason to use an autoencoder. Ideally, you would like to reduce the dimensionality of your data before feeding it into a fully connected neural (FCN) network, as otherwise the number of parameters would be unwieldy (particularly for images). 

The below model uses convolutional layers in the encoder to compress an MNIST image down to a given latent dimension. It is then blown up to the original dimensions. Even using a relatively small latent dimension of 20, results in remarkable recreations. This encoded image can then be fed into a classification network, and may work better than using the original image. 

**TensorFlow Example of a Convolutional Encoder with a FCN Decoder**
```python
class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
	# calls __init__ of Model superclass
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        Conv2D(32, (3,3), padding="same", activation="relu"),
        MaxPool2D(),
        Conv2D(32, (3,3), padding="same", activation="relu"),
        MaxPool2D(),
        Flatten(),
        layers.Dense(latent_dim),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# Note: You can still access the individual models
encoded_imgs = autoencoder.encoder(inputs).numpy()
decoded_imgs = autoencoder.decoder(encoded_inputs).numpy()
```
#### 2. Denoising
Autoencoders are particularly useful for denoising. That is, we take a noisy image and clean it up by removing the stuff that doesn't matter (in this case Gaussian noise). 

**An example of the effectiveness of a denoising autoencoder**
![[autoencoder_denoising.png]]

For these tasks, I've personally noticed that a fully convolutional autoencoder works best. Moreover, it is also more generalizable. I discovered this by training a network on MNIST but then having it de-noise samples from KMNIST. Note that this was most likely not truly Gaussian noise in the KMNIST samples and these were not present in the training set. Similar attempts with a convolutional encoder and a FCN decoder failed to recreate the KMNIST samples. 

**Convolutional Autoencoder Trained on MNIST Generalizing to KMNIST**
![[autoencoder_kmnist.png]]

**TensorFlow Example of a Convolutional Autoencoder**
```python 
class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        Conv2D(32, (3,3), strides=1, activation="relu", padding="same"), 
        Conv2D(32, (3,3), strides=2, activation="relu", padding="same"),
        Conv2D(32, (3,3), strides=2, activation="relu", padding="same"), 
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"), 
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=1, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"), 
        Conv2D(1, (3, 3), activation="sigmoid", padding="same")
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
```
#### 3. Anomaly Detection
First, let's think what an encoding actually is. It is some vector within the latent space that represents some object that the neural network encountered while training. If we pass an input into the encoder, which is different than samples in the autoencoder's training set, we should not expect the encoder to know how to handle this different sample. 

In fact, it will not have an encoding for this different sample that makes sense. We can use this to find anomalies. If an input is an anomaly and not contained in the training set (say a suspicious credit card transaction), then the distance of its encoding will be large from non-anomalous inputs. 

![[autoencoder_anomaly.svg|550]]






