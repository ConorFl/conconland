### **Building makemore Part 2: MLP**

[source](https://www.youtube.com/watch?v=TCH_1BHY58I&t)

Models 1 and 2 only considered a single previous letter in a word (name).

Bengio et al is canonical paper on the topic.

Side note on .view(…) and storage in PyTorch: PyTorch internals blogpost explains that using .view(…) is extremely efficient as compared to unbinds and concats because tensors are actually just stored as one long array, and using .view(…) just changes how it is spit out, where as the other options actually restructure or create a new tensor (using new memory). .view(…) can also infer a dimension if you pass -1 (ex: emb.view(16, -1)).

For calculating losses, just use F.cross_entropy, don’t actually calculate logits (log counts), raise to e, etc., it’s more efficient forward and backwards and solves some edge cases.
### **Mini-batches**

Going through the entire dataset for every single weight update takes too long. Instead let’s pick a random smaller batch.

**Note:** because we are only uses a small set, our gradients are less reliable, however they are usually directionally right. They create a more noisy loss function (thick up and down thrashing, but generally headed the correct way).
### **_How to pick learning rate?_**

One option: try a wide range (ex: 0.0001 to 1) apply them to the same model and store all the losses. The shape should be high loss (not learning fast enough), big dip low (good learning), then thrashing all over (too high, unstable). You want something in the dip.

Eventually, you’ll probably want to decay the learning rate by factor of, say 10.

### **Let's talk Train, dev/val, test**

These losses are misleading, because they are on the train data. They could be overfitting, which becomes more likely as the model grows in complexity. Instead we should split into:

- train: for training
- dev/val: for selecting hyperparams
- test: final evaluation 

We should only touch test sparingly. Any time we use test, we start to risk learning something from it.

From here, we train a bunch of times, and see that train and dev are still staying very close. This suggests **underfitting**, and that either we need to train more or the model simply isn’t complex enough. To address this, we explore different ways to change the model: larger hidden layer(s), larger embedding, different batch sizes, expanding how many chars back we look as an input, etc. 

  The full formal process would be experimenting on all these hyper-parameters using the dev set, then reporting final data on the test set.
### **Questions:**
- The letter to 2D embedding C is used 3 times in the front of the network. How does back prop work with that? How is one grad calculated and updated? Maybe answered in an earlier video?
	- Yes. Gradients are stored on each node and **accumulated/summed a**s backprop occurs. See “**Tada! Backpropagation!**” in Lecture 1.
### Todo:
- [x] beat the number he got as 1:12:00.
- [ ] Read Bengio paper.