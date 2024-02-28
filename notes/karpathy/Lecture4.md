# **Building makemore Part 3: Activations & Gradients, BatchNorm**

source: <https://www.youtube.com/watch?v=P6sfmUTpUmc>

We want to look at RNN’s eventually, but let’s hang back a little longer and learn more about activations and gradients. This will also help us understand what makes RNN’s tricky to optimize.

We start by cleaning up the code for last session’s neural network (removing magic numbers, etc.).
### **Initialization**

Let’s start by considering initialization. We see REALLY high initial loss (~27) that eventually shoots down rapidly (hockey stick). This is very bad. It’s not just wrong, it’s **_confidently_** wrong, outputting very high logits for wrong answers.
### **_What would be a reasonable “starting error”?_**

A reasonable starting error depends on the problem, for us, it would be the NLL when all outputted probabilities were equal. This is closer to 3. Note, any equal value (for all 27 logins) would work given how softmax works. Simplest way to do this would be to target all 0 logits to start. logits are W2 * h + b, so we can just make the b’s all 0, and the W2’s very small.
### _Can we make W2 exactly 0?_

It would probably be fine here, but generally no (discussed later).

Now we see better results on test data (because we no longer waste the beginning cleaning up horrible weights).
### **Saturated tanh’s (activations)**

We look at the tanh’s for a batch (32 in batch, 100 hidden layer outputs -> 3200 values) and it’s almost all -1 and 1. Why? The distribution of the inputs / pre-activations (W1 * x + b1) is a _wide_ bell curve. All the spread out values gets squashed by tanh into -1 or 1.
### **_Is this bad?_**

Yes! :/ When we try to do backprop with (dh / dw1) * (dL / dh) for some w1 in W1, dh / dw1 will be essentially 0 since any moves will still be far along the flat portion of tanh. Therefore w1 = w1 - lr * (dL / dw1) = d1 - lr * (0) and no more learning will occur, the gradients will have **vanished**. Similar problems can happen with ReLU neurons (either due to initialization, or because some very high learning rate overshot and caused it to always return 0, aka brain damage).
### **_How can we fix?_**

We want the pre-activations (W1 * x + b1) to be closer to 0. We do similar to what we did to shrink logits. Multiple W1, b1 by small numbers. Now we check the distributions, and the activations (tanh’s) look much better (not two single peaks at -1, 1, more like a bell curve or two peaks but connected like a skate ramp) and the pre-activations are a bell curve but MUCH less wide.

_For b1, 0 or 0.0001, why either or?_ Having 0.0001 can add a little bit of diversity which in practice he says can work well. 
### **Expanding this strategy to deeper networks**

**RECAP:** Random output weights can cause the network to be confidently wrong, and waste early training, so we scaled them down. Random hidden layer weights can cause gradients to vanish, so we scaled them down.

Our network is small though, so even with the original bad initializations it was able to learn, but this probably wouldn’t work for deeper networks. The vanishing gradients would compound with more hidden layers. We could keep multiplying weight initializations to keep them small, but where did these multiples 0.001, etc. come from? Not very principled approach.

We want approximately unit Gaussians throughout the NN (why? Because if too low just -1 activation and or too high +1), but Var(W1 * x + b1) won’t be unit Gaussian if W1 is unit Gaussian, so again how _exactly_ do we scale?

Kaiming et al found that gain / (fan_in ** 0.5) works well as the standard deviation, where gain is some factor depending on your activation function (ex: for ReLU, it’s 2 ** 0.5 since you threw away half of your distribution). We also want _gradients_ to be well behave. Kaiming He et al, found if you properly initialize the forward pass, the backward pass will be approximately good (multiple by some factor).

PyTorch has Kaiming initializations available where can specify your activation function (which determines the gain) and whether to optimize for forward pass (the default) or back pass.
### **_Why do we need these “gain” things?_** 

The activation functions squash (tanh) or eliminate (ReLU) outputs. We need to fight the squeezing a little bit to get it back to unit Gaussian-ish. See next section for more info.
### **_Wait, how do we know the pre-activation will be unit Gaussian if we’re only adjusting W’s, but can’t control x?_**

It seems like we are assuming x is unit Gaussian (from either normalizing inputs or correctly initializing earlier weights), and the goal of this is to pick W that _preserves_ the unit Gaussian-ness of x.

**Note:** a lot of these have become less important to precisely initialize your NN correctly due to things like ResNets (discussed later) and things like…
### **BatchNorm**
  
If we want unit Gaussian pre-activations, why don’t you just normalize them within the batch: `hp_norm = hpreact = (hpreact - hpreact.mean()) / hpreact.std()`. However, we only want unit Gaussian _at the start_. During training, we want to be able to move, so we introduce gain and shift which the model will learn: `gain * hp_norm + shift`. Note: gain and shift are shared and learned across all batches (while the mean and std used are batch specific). We initialize them to gain = 1, shift = 0, which combined with normalizing, means we start with… a unit Gaussian.

We generally put the BatchNorms on the linear layers (the pre-activations), or convolutions.
### **_Personal sidenote: didn’t we already just go through normalizing weights with Kaiming? What’s different here?_**

We only normalized the weights to make sure weights aren’t the cause of over high pre-activation values, causing over saturated activations and vanishing gradients. But recall pre-activations are W\*x + b, we can’t control for x, so the total pre-activation could still be too high or low. This is why we must normalize again (and why getting initialization right isn’t as important if we’re just going to normalize the full W\*x + b pre-activation anyway).

ACTUALLY, this was my initial read, but I don’t think this is true after re-reviewing and adding the **_Wait, how do we know_** section.

I think the actual difference is that Haiming, etc. helps with initialization, preventing dead neurons, etc. but does nothing during continuous training, that’s why we have BN.
### **_Wait, during a forward pass, each element in a batch is now of function of the other examples in a batch. Isn’t that weird?_**

Yes, that’s weird and means the output during training for a given sample will jitter depending on what’s in its batch. The jitteriness introduced is actually sorta good because it creates noise which is almost like added augmented data. So even if we wanted to remove BatchNorm since the batch dependencies are weird and it's error proned, it’s difficult to remove because it has secondary benefits. 
### **_What about using the network for inference after training? There’s no more “batch”?_**

Yes, that is a problem. One solution is to separately just calculate the mean and std of hpreact across the whole training set and use that. Having this secondary task might be a little annoying. An alternative approach would be the try to calculate a running mean during the forward passes:

- running_mean = 0.999 * running_mean + 0.001 * batch_mean
- running_std = 0.999 * running_std + 0.001 * batch_std

This gets pretty close to the true answer and avoids a secondary task.
### **Two other details about BN to notice:**

1. In the paper, when normalizing they do (x - mean) / (var + epsilon) ** 0.5, what’s epsilon for? Just some small number guarantee we don’t divide by 0 (since var in non-negative).
2. The pre-activations (W\*x + b) have a bias b. And when we normalize, and  subtract the mean, they get subtracted out, so they’re pointless and can be removed. It’s not hurting us, it’s just pointless. The biasing is done by the BN shift variable.

### **Walks through ResNet50 to explore:**

- The BN’s in there after convolutions. As a reminder, convolutions are the same as these linear layers of matrix multiplication, it’s just a linear layer with a small W over overlapping sections.
- nn.LinearLayer(in, out) and shows that weights get initialized such that if you have roughly Gaussian input, you’ll have roughly Gaussian output (very similar to what **Initialization** stuff from earlier).
- nn.BatchNorm1D: This has a momentum parameter, which is used for the running mean / std (default is 0.1, we used 0.001 above). For small batch sizes (like 32), we need a small momentum otherwise we’ll see big swings.

Full recap at 1:14:00.

Followed by pytorch-ifying our code. Making these layers as \_\_call__able classes.

### _Done... Just kidding let’s do more… specifically look at some **diagnostic tools**._

**Viz 1: Forward pass activation statistics**

To better appreciate the impact of initializations, we look at a network with multiple tanh layers. We scale them by 1 / fan_in ** 0.5. We then try with a **gain** of 5 / 3 (nice consistent std’s), with 1.0 (they shrink as we move forward since tanh squashes), and with 3 (they explode and we’re back to all -1 and 1 activations or **saturated** activations).

**Viz 2: Gradient statistics**

We play the same game of messing with the gain. Ideal scenario means they all have similar distributions, whereas too big or small cause the distributions to be different at different layers, which is bad.

_Why do we need the non-linear activations again?_ Otherwise it’s just linear regression. We also look at this with no tanh (all Linear). In this case the correct gain is 1.0. Also this is just a linear regression… HOWEVER backprop ends up working differently from just a single linear layer.

**Viz 3: Parameter activation and gradient statistics** 

We now look at the grad:data ratio (the gradient compared to the actual weight). This should be very small, otherwise you’re going to swing the weights around. In our example, the last layer had significantly higher. We also plot lr\*p.grad.std() / p.data.std().log10(). This should be ~1/1000, but it varies by layer. Below would mean not training fast enough, too far about means too much training.
### **Finally, let’s add BN to makemore…**

And now everything works well and feels more stable, less brittle. For example, if we change gain to 2, or remove fan in, it still mostly works (though you might need to change your learning rate, as observed in viz 3 moving up or down from 1 / 1000).

Our current bottleneck is probably context length (last 3 characters), therefore we need something like RNN’s or transformers.

### **TODO:**

- [ ] Read paper Kaiming He et al.
- [x] Read Batch Norm paper