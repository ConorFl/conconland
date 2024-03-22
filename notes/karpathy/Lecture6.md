# **Building make more Part 5: Building a WaveNet**

[source](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)

So far, makemore is a fairly simple NN with a single hidden layer. It takes 3 characters in as input. To improve the model, we want to:

1. Expand beyond 3 characters.
2. Stop squashing all the information into a single layer. Instead we want a deeper model that fuses the information little by little.  

### **But first, some code clean up**

- The loss chart doesn’t have the hockey stick anymore, but it’s still very thick, due to the fact that are batches are very small, causing high variance in the loss. To fix this, we average a bunch of consecutive losses. The chart is much more readable, and it’s also easier to see a noticeable drop when the learning rate changes.
- Turn EVERY part of the model into PyTorch-like layers. We already did Linear, BatchNorm, and Tanh. We now add a layer for the embedding of letters into C, and flatten (needed since the embedding is used 3 times). With all of this done, we can now also create a PyTorch like Sequential to build our model.

Ok, now it’s easier to retrain the model with a context window of 8 characters (no magic numbers like 3 sprinkled into the model). The length 8 context window sees significant improvement of length 3, but this is just a baseline before we get into…
### **WaveNet**

The idea behind WaveNet is to combine the embeddings little by little. With one flat layer, we can only identify linear relationships between different characters in the embedding.

In our current (length 8 context) model, we concatenate all 8 characters:
```
(1 2 3 4 5 6 7 8)
```
Instead, we want to concatenate only 2 at a time.
```
(1 2) (3 4) (5 6) (7 8)
```
and run all of them through the same weights (a convolution).

**Math sidenote:** so far, for matrix multiplication we’ve seen shapes like (a, b) @ (b, c) = (a, c). It doesn’t need to be 2 dimensions, as long as the touching dimensions are identical (ex: (a, b, c) @ (c, d) = (a, b, d)).

Our old hidden layer outputs 200 neurons, so the dimensions change from: (32, 80) @ (80, 200) to (32, 4, 20) @ (20, 200)
### **_The number of weights in this layer shrunk significantly (80 x 200 -> 20 x 200), how does that work?_**

The (20, 200) weights are going to get used **_4 times_**, once on each pair (i, i + 1). This is a **convolution**.

Going over the model, and the printed out dimensions of the output layer is a **very** helpful way to test yourself.

### **_Did adding this convolution that re-uses weights and makes our outputs 3 dimensional break any of our code? Yes!_**

The BatchNorm layer is now broken. The Linear layer gets used 4 times to output (32, 4, 20) @ (20, 200) = (32, 4, 200). But for each of the 32 rows, the 4 sets of 200 were calculated using the _same_ 20 x 200 weights of the Linear layer. So for tracking things like mean and var, these should be collapsed. Our BatchNorm layer doesn’t know this, so it tracks them separately. This is not a huge deal, but the mean and variance will be more reliable if we quadruple the number of values to calculate them.

With this bug fixed, we see a slight improvement over the flat length 8 context model. By increasing the size of the model, we see a larger improvement in loss.

The final observation is that we have improvements, but we don’t have an “experimental harness” to structure how to improve our model. We are just doing a guess and check, and it’s taking longer as the model gets more complex.
### **_So is this a convolutional neural network?_**

Not exactly, due to the inefficiency in how we implemented it. Think of our training data:
  
```
*******d -> i

******di -> o

*****dio -> n

****dion -> d
```

Each of these will pass through the network as a different training example. That means `(di)` will go through twice (row 2, 4, etc.). This is inefficient. In a CNN, these inefficiencies are addressed. You can think of a CNN as like a for loop sliding over the full name Diondra.
