# **Building makemore Part 4: Becoming a Backprop Ninja**

[source](https://www.youtube.com/watch?v=q8SA3rM6ckI&t)

This video doesn’t lend itself to these conversational notes. In this video, the goal is to do `loss.backward()` by hand and calculate the gradients (dloss/dw) for every intermediate w.

The idea is that back in the day (2012) people needed to do this and something is lost by not going through the exercise.
### Some key takeaways:

- If a forward pass involves broadcasting a tensor (say across a row or column), the backprop will probably involve a sum (since the tensor was reused a bunch of times.
- Vice versa (summing in forward pass implies broadcasting when calculating gradients in backprop).
- The softmax’ed probabilities sum to 1.0 (obviously). The gradients on the probs are basically identical, except the correct answer had a 1 subtracted (so the gradients sum to 0). If the model made a perfect prediction (1.0 output for the right answer, they would all be 0’s).
- When trying to handroll gradients for W of size (n, m), you know dloss/dW will have the same dimensions, and you have a few candidates as to what the derivative will be made up of (the outputted matrix, the matrix W is multiplied by, etc.), so from there you can sorta just look at the dimensions of your candidates and which ones would combine to produce at least the right dimensions.
- **Bessell Correction:** The sample variance is actually a biased estimate of population variance. It under estimates the true variance (`(n - 1)/n * true_var`). It still feels a little fuzzy, but I think the idea is that the sample variance is a little closer to the sample mean (since it’s made of it), so we need to multiple by `n / (n - 1)`, which simplifies to `1 / (n - 1) * sum(x - sample_mean)^2`. [https://math.oxford.emory.edu/site/math117/besselCorrection/](https://math.oxford.emory.edu/site/math117/besselCorrection/) 

We finish by taken some of the line by line derivations calculated earlier, and simplifying them drastically.