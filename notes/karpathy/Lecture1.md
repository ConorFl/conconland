# The spelled-out intro to neural networks and backpropagation: building micrograd

We’re going to build an **Autograd** (automatic gradient) engine to perform backpropagation (reverse-mode autodiff), which allows you to efficiently evaluate the gradient of some loss function, with respect to the weights of a NN. This allows us to tune the weights to minimize the loss.
### **_What’s the big deal about this little library?_**

The big deal is that we can build a graph (ex: L = a * b + c - b, and then run L.backward(), which will calculate dL/da, dL/db, …). These gradients are how NN weights get tuned to minimize loss. 

**Remember:** a NN is just a mathematical expression.

The first 20 minutes or so are used to review what a derivative is, and what a partial derivative is (ex: what _exactly_ is dL/da from above?).

We next start building out Value objects. We don’t want a, b, c, L, to just be integers, we want Value objects. 
### **_Why? What’s wrong with just ints?_**

The usefulness will become more apparent over time. We want to maintain a graph of how each number is calculated (the two “children” and operation that created the Value object), and later, each Value will have a gradient instance value that will be important for backprop and tuning.

The next few minutes build out Values by adding dunders  __add__, __mult__, adding labels and visualization.
### **_Let’s start backprop (manually)_**

dL/dL is just 1.0.

Then we revisit the calculus. Notably:

- For L = d * f, dL/dd = f and dL/df = d
- For d = c + e, dd/dc = dd/de = 1.0
- dL/dc = dL/dd * dd/dc (Chain Rule)

Some interesting observations when viewing the graph:
- For addition in the graph, the local derivative will be 1.0 (c = a + b, dc/da = 1.0), so dL/dc = dL/dd * 1.0, therefore we basically just push back the gradient from the sum Value node.
- For multiplication, the local derivative of one factor is the other (c = a * b, dc/da = b). So the gradients are just the factors swapped, times the gradient of the product Value node.

### **Moving from adding / multiplying to neurons**

Pretty much the only operating missing on our Value class is an activation function like tanh, so we implement a tanh method.

We now manually calculate all the gradients by looking up the derivative of tanh (1 - tanh^2) and using the “interesting observations” from above.
### **Automating gradient calculations**

For each op (+, \*, tanh), we created a new Value instance (out) and return it. We must now also codify how to calculate each gradient. Specifically, we define a function \_backward on out, which has a reference in scope to the two Value nodes used to create out, and sets their grad values appropriately.

Finally, we set the final node’s gradient to 1.0 (dL/dL = 1.0 by default), then call \_backward() on all children is reverse topological order.
### **Tada! Backpropagation!**

Major bug: when a variable is used more than once (ex: a + a = b, or a lookup table used later in lecture 3), the gradients are getting overwritten. Multivariate calculus tells us we should just be _accumulating_ the gradients.

Around 1:20:00 this overwriting error is shown. 

This does require us to make sure we reset grads to 0 after each pass. Around 2:10:00, the error from forgetting this is shown. :)
### **_Let’s see what this whole thing would look like in PyTorch_** 

Around 1:40:00, the whole this is redone in PyTorch. Notably, we need to specify requires_grad = True on all leaf nodes because normally they don’t have it for efficiency (it’s usually the input values).

It also has a .backward() function like our micrograd.
### **_How is it actually different from micrograd?_**

PyTorch works with n dimensional **tensors** instead of scalar Values. However the math is all the same. Everything else is mostly optimizations.
### **Implementing NN helpers in micrograd**

Finally, we implement:

- a Neuron (a collection of _nin_ weight Values, a bias b, an activation fn. and a call dunder)
- a Layer (a collection of _nout_ Neurons with _nin_ inputs each, and a call dunder)
- a Multi-Layer Percentron or MLP (a collection of layers were with specified input/output size for each Layer, including the output layer, and a call dunder)

### **_What next? What do we do with these gradients?_**

Up to now, we’ve explored what gradients are, and how to calculate them, but we don’t know what to do with them.

Ultimately we want our model to make good predictions. To evaluate the model, we want to define a single value to assess it: a loss. We use the sum of squares of true vs predicted values, then call backward() on that.

Now every node (even input nodes :/) has gradients with respect to loss. We want to gather all the weights and nudge them by subtracting their gradients.
### **_Why subtraction again?_**

The gradient of w is dL/dw. If dL/dw is positive, it means moving this number w up would increase the loss, we want to move it down so we subtract our positive gradient. If dL/dw is negative, moving w up would decrease the loss, we want to move it up so we subtract our negative gradient.

This process of repeatedly updating weights in the opposite direction of the ever changing gradients is called…
### **…gradient descent!**

The rest of the lecture is just cleanup like implementing a loop for backprop and gradient descent.
### **Important takeaways** 

When I first did PyTorch it wasn’t clear why I was doing all this hand rolling, instead of just model.train().

The key thing to realize is the backward() is only re-calculating the gradients at each node. That’s why I also needed to update weights.
### **Questions**

- The question I always have: why can we update all weights at once? When we update layer n, then update layer n-1, we’re using stale gradients for layer n, no?
	- My guess is, even if the grad is stale, it’s still directionally correct, so it mostly works.

### **Todo**

- [ ] Ask question on YouTube.
- [x] He mentioned a discussion forum! At the very end.
- [x] Go over the extra classification notebook on GitHub.