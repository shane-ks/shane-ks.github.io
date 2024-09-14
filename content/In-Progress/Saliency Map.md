### Introduction
A <mark style="background: #BBFABBA6;">Saliency map </mark> is a way to measure the spatial support of a particular class in each image by using a <mark style="background: #D2B3FFA6;">local gradient-based [[Backpropagation]] interpretation method</mark>. 

For example, suppose we have an image of a dog that we will pass to a dog and cat classifier. We would like to ask the question, "what are the pixels responsible for classifying this picture as a dog?"

**The pixels brighter below have a greater effect on the classification as a dog**
![[saliency_map_dog.png]]

It is the oldest and most frequently used visualization method for interpreting the predictions of a [[Convolutional Neural Network]].
### Methods of Saliency Maps




### Limitations
- Not always reliable. Indeed, subtracting the mean and normalizations, can make undesirable changes in saliency maps as shown by https://arxiv.org/abs/1711.00867
- Saliency maps are vulnerable to adversarial attacks https://arxiv.org/abs/1710.10547
- https://papers.nips.cc/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf tested many saliency maps techniques and found that GradCAM and gradient base are the most reliable.
