### Introduction
The occlusion method runs a patch over the image to determine which pixels affect the classification. That is, we *occlude* patches of the image. This is a method to help visualize what your model is actually doing. The occlusion method is a [[Forward Pass Attribution Method]], as compared to a [[Saliency Map]], which is a [[Backward Pass Attribution Method]]. 

**Splitting image into occlusions and then constructing image of losses**
![[saliency_map_occlusion.svg|580]]
### More closely examined
More rigorously, suppose we have some classification neural network and some training image. We take note of the true label $K$. We then get the prediction of the original image, $Q$. We denote the loss without occlusion as $L_\mathrm{nooccl} = -\log P(y = Q)$. 

We now occlude patches of the image with gray blocks starting at the top left and get the predictions of these occluded versions. The loss of the $i$-th occlusion is then $L_{\mathrm{occl}_i} = -\log P_{\mathrm{occl}}(y = Q)$. So, the change in the loss for the $i$-th occlusion compared to the original is 
$$
\delta L_i = L_{\mathrm{nooccl}} - L_{\mathrm{occl}_i}
$$
which we then use to construct the image on the right in the previous diagram. 
$$
\mathrm{Occlusion\,loss\,map} = \begin{bmatrix} \delta L_i
\end{bmatrix}_{i=1}^n \rightarrow \mathrm{reshaped\,to\,starting\,dimension}
$$
For example, if there were 4 inclusions, then the resulting occlusion loss map would be 
$$
\begin{bmatrix}
\delta L_1 & \delta L_2 \\ 
\delta L_3 & \delta L_4 \\ 
\end{bmatrix}
$$
Note that if $L_{\mathrm{occl}_i}$ is much smaller than $L_{\mathrm{nooccl}}$, then the $i$-th occlusion had a significant impact on the classification. This means that when the quantity $\delta L_i$ is large, then we know that the pixels contained within the $i$-th occlusion contribute significantly to the original image's classification. 

When $K = Q$, we use this saliency map to determine which parts of the image contributed to the correct predictions. However, if $K \neq Q$, we use it to determine which parts of the images contributed to predicting the incorrect class. 