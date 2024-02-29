# The spelled-out intro to language modeling: building makemore

[source](https://www.youtube.com/watch?v=PaCmpygFfXo&t)

makemore is a model that tries to generate new “human-like” names from scratch (ex: Jilliam might be a name it makes up). It is a character level language model (as opposed to a word level like Chat-GPT). 
### **_How exactly does a model “generate” new names? What are the inputs/outputs?_**

The input is some part of the name so far (ex: the last character, which could just be `<START>`), the output is a distribution of the likelihood of all 26 letters (and `<END>`) to come next, given the input. This distribution is **sampled** to get the next character. The sampling is why it can produce multiple different answers and there isn’t just one “right” answer with the most likely char.

We will do this using a number of approaches:
1. Bigrams: Track the frequency that letters appear next to each other to build an output distribution: P(letter X \| letter Y just appeared)
2. Bag of words?

### **Model 1: Bigrams**

Go through each name, break it up into pairs of characters and count frequencies. This produces a 27x27 grid of counts. The rows are each letter, and the cols are the frequency that the col’s letter appeared after the row’s letter. Dividing each row by the row’s sum turns the row into a distribution (sums to 1).
### **Sidenote on Broadcasting Rules / Semantics**

1. Align dimensions on the right.
2. Each dimension must be equal, 1, or not exist.

You still want to make sure it’s doing what’s you’re doing (ex: copying when dim is 1 or doesn’t exist). Take broadcasting seriously.
### **_How do we evaluate the quality of this model with a single number?_**

The way we measure quality is by considering “how likely” is this name? And we have already calculated the likelihood of the first letter, and the second letter after the first letter, etc. We can multiply all of these, and the product is the likelihood.

These can get kinda small, so it can be easier to deal with the log likelihood, since log is a monotonic function and log(a\*b) = log(a) + log(b).

Finally, log from \[0, 1] maps to -inf. to 0. If we negate this, it goes from inf. for very low, near 0 probabilities, to 0 for very high, near 1 probabilities. Thus, we use **negative log-likelihood** as a loss function.

We may also divide by the number of pairs of chars to normalize loss.

Max likelihood ~ max log likelihood ~ min neg. log likelihood ~ min avg neg. log likelihood.
### **_What happens if you evaluate the likelihood of a word with a pair of chars that didn’t appear in training data?_**

Neg. log likelihood will be inf. One way to get around this is by **smoothing** your model. That is, add 1 (or more) to every count, so everything appears at least once. The more you smooth, the more your model approaches equal probabilities.
### **Model 2: Neural net**

We are going to reach an almost identical conclusion, except we will do it using a NN. Here’s the plan: 

1. Input a character.
2. Output a distribution of the next character (27 possible values).
3. Check the likelihood the model assigns to the “true” next character from the training data.
4. Calculate the neg. log likelihood (NLL) as above, and perform stochastic gradient descent to try to make the correct probabilities higher, which will lower NLL.

### **_How do you input these single characters into a NN?_**

First, we map them to int with our stoi mapping. Then we use **one-hot encoding** where we map integer x to an array of 27 0s, except for the x-th index, which is 1.

  The whole network will just be W, a 27x27 hidden / output layer. With the one-hot encoding, each row basically maps to a row from the bigram counts in **Model 1**.
### **_How do we turn these outputs into a probability distribution?_**

Think of them as **log-counts** (or **logits**), to turn them into regular counts, raise e to each of them (turning them all positive), then divide by sum(counts). This is called **soft-max**.
### **_Wait, but how do we actually do back prop?_**

From here, we just do back prop like in micrograd: Pluck out the probability for each of the true answers, calculate loss as NLL, calculate dL/dw’s, loss w.r.t. each weight in W, working backwards, and adjust them to minimize loss.

Remember, because the final outputted loss is calculated using PyTorch, it’s not simply a single number, it’s a graph of all the calculations that made up the number. And each node is the graph has a grad value, which is specifically how we do back prop by just calling loss.backward().
### **_What kind of performance (i.e. NLL) do we expect?_**

Identical to bigram (lol), because we haven’t taken in any other data. We’re basically just recreating it using NN instead of counting. In fact, when we sample, we get the same sample names. Probably because the distributions are so similar that rand() translates to the same letter in the NN output distribution as it did in the bigram distribution.
### **_With bigrams, we had smoothing, is there an equivalent for NN?_**

Yes, we can add **regularization** to our loss function, where we add the w weights squared to our loss function (times some small constant), which will incentivize weights that don’t get too big.