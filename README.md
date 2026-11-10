# Lab 1-3: Transformer language models

## Goal

In this lab you will upgrade your attention-based language model into a Transformer, including an implementation of all the additional pieces that make a Transformer: positional embeddings, residual connections, normalization and an MLP block.
You will build 3 types of Transformer models: a model made of a single Transformer block; a stack of blocks; and a stack of multi-head attention Transformer blocks.
You will train all three model types on the TinyStories corpus, compare them against your Attention-LM results, and search for the best potential model settings.

Before the lab, you should read [chapter 7.2](https://web.stanford.edu/~jurafsky/slp3/7.pdf) from Jurafsky's "Speech and Language Processing", or watch one of the million introductory videos on Transformer models from YouTube.
If you're brave enough, you should read [the paper that introduced transformers](https://arxiv.org/abs/1706.03762), which is arguably one of the most important papers in deep learning in the last decade. Do note that the paper presents an Encoder-Decoder architecture, while this course will be focused on the decoder-only architecture. For more information, see [this link](https://huggingface.co/learn/llm-course/en/chapter1/6).

## Setup

You have been given multiple files:
```
 agents.md                    # A guidelines file for coding agents - keep it included with the code at all times
 README.md                    # This file, containing instructions and questions
 lab-1-3-transformer-lm.py    # The code file which you will edit
 test_1_3.py                  # A test file to run after each TODO for validation
```

The tokenizer (`TinyStoriesBPETokenizer`), `batched_causal_mask` and the training, evaluation and generation code are all given. You do not need to edit them.

**Note:** Before you start, copy your implementations of `causal_mask` and `attention` from lab-1-2 into `lab-1-3-transformer-lm.py`. The code in this lab depends on them.

To run the test file, run it as a python script, either with TODO numbers or without arguments to run all tests:

```
python test_1_3.py 1      # check TODO 1
python test_1_3.py 2 3    # check several TODOs at once
python test_1_3.py        # check everything
python lab-1-3-transformer-lm.py
```

The tests run in a few seconds.
Note that these tests are basic and non-comprehensive.
The exercise will be graded using unseen tests.


## TODOs

The TODOs are marked `# TODO COMPLETE (N)` in the lab file. Each one is specified by the docstring of the function it sits in, so read that docstring before you write anything. Do them in order: later TODOs build on earlier ones.

### TODO 1: Positional embeddings

The first part of the transformer you will implement are its positional embeddings: the added representation of the position of each token in the input sequence.
On its own, attention has no signal of a token's position in the sequence, so if you shuffle the context every attention weight will remain the same. Positional embeddings are what tell the model about order.
There are MANY ways to implement positional embeddings, but the original transformer paper builds them from sines and cosines of the position (called *sinusoidal positional embeddings*):

The positional embedding of time step $t$ is a vector of $D$ elements. Each element is the value of a sine or a cosine at a phase set by the element's index $i$:

$$
pe(t, i) =
\begin{cases}
\sin\left(t / 10000^{i/D}\right) & \text{if } i \text{ is even} \\
\cos\left(t / 10000^{(i-1)/D}\right) & \text{if } i \text{ is odd}
\end{cases}
$$

The base $10000$ and the division by $D$ are empirical choices.

Implement `get_positional_embeddings` to apply this positional embedding, and test your implementation:

```
python test_1_3.py 1
```

The given `PositionalEmbedding` module will use your implementation to add the positional embeddings to the token embeddings, for a position-aware representation of the embeddings.

### TODO 2: A single Transformer block, inline

Now that we have the positional information ready, let's combine it with attention (and the rest of the transformer components) to a single transformer layer (in the `TransformerLM` class). 

Start by defining all of the parts of a transformer layer in `TransformerLM.__init__`: the positional embedding (using the `PositionalEmbedding` class which relies on your implementation), the attention layer's query/key/value projections, an MLP block (read about `torch.nn.Sequential`, `torch.nn.Linear` and `torch.nn.ReLU`) and the layer normalization layers (you can use the ready-to-use implementation of `torch.nn.LayerNorm`). You will also need to define the final projection of the context to the vocabulary (as you did in the previous lab).
Replace each `...` in `TransformerLM.__init__` with the layer it should hold, and keep the variable names.
      
Then, implement a forward pass through a transformer block in `TransformerLM.forward`: remember to include all parts you read about (including the positional embedding, a batched attention call (using the given `batched_causal_mask` function and your `attention` function) and an MLP forward pass), the layer normalizations, and the residual connections. In this lab we'll remain loyal to the original transformers paper and put the normalization AFTER connecting the residual with the output of the attention/MLP blocks: 

$$
\begin{aligned}
X &= \text{PositionalEmbedding}(\text{input\_ids}) \\
Q &= X W_Q + b_Q, \quad K = X W_K + b_K, \quad V = X W_V + b_V \\
A &= \text{Attention}(Q, K, V, \text{mask}) \\
H &= \text{LayerNorm}_1(X + A) \\
\text{MLP}(H) &= \text{ReLU}(H W_1 + b_1) W_2 + b_2 \\
Y &= \text{LayerNorm}_2(H + \text{MLP}(H)) \\
\text{logits} &= Y W_{out} + b_{out}
\end{aligned}
$$

Here $X$, $A$, $H$ and $Y$ have shape `(batch_size, seq_len, hidden_size)`, and `logits` has shape `(batch_size, seq_len, vocab_size)`.


Test your implementation:

```
python test_1_3.py 2
```

And once it passes, train this single-layer transformer model:

```
python lab-1-3-transformer-lm.py --model single
```

Did you get better validation perplexity compared to the results from the previous lab ("No" is also a possible answer)? Hypothesize why.


### TODO 3 & 4: Stacking blocks

One layer is nice, but not enough. We often want to stack a few layers so that our models are able to model more complex behaviors.

Now you will implement a multi-layer (or multi-block) transformer model.
To stack layers, implement `TransformerBlock` as a reusable module: Implement `TransformerBlock.__init__` and `TransformerBlock.forward`, which define and run a forward pass through a single block (which expects a batch of hidden vectors as input and returns an identically-shaped batch as output). These should be very similar to your implementation from the previous part. 
Note that this block doesn't include positional embeddings (which are added only **ONCE** for each token, before the first block) nor the output vocabulary projection.

Afterwards, implement `MultiLayerTransformerLM.forward`: convert the input tokens to embedding vectors, pass them through the transformer blocks list, and project them to the output vocabulary.

Test your implementation, and once it passes train the multilayer model:

```
python test_1_3.py 3 4
python lab-1-3-transformer-lm.py --model multilayer --num-layers 2
```


### TODO 5, 6 & 7: Multi-head attention

The multilayer model was better, but its still limited in one large aspect: each layer computes a **single** attention pattern (or attention distribution) per position. 
The simplest way to solve this is to allow multiple patterns per layer. In formal language, this is what's called "multi-head attention".

In **multi-head attention**, multiple attention heads are computed in parallel at each layer, each with **its own** query, key and value projections. 
This is done by splitting the hidden vector into `num_heads` smaller vectors and attending separately within each one, then glueing the results back together.

In practice, we don't create a separate linear layer per head. Instead, one linear layer of size `hidden_size` → `hidden_size` computes the queries (or keys, or values) of all heads at once, and each head takes its own slice of the result. Each head has size `head_size = hidden_size / num_heads`. For $H$ heads, multi-head attention computes:

$$
\begin{aligned}
Q &= X W_Q + b_Q, \quad K = X W_K + b_K, \quad V = X W_V + b_V \\
Q_h, K_h, V_h &= \text{the } h\text{-th slice of size head\_size of } Q, K, V \\
\text{head}_h &= \text{Attention}(Q_h, K_h, V_h, \text{mask}), \quad h = 1, \dots, H \\
\text{MultiHead}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O + b_O
\end{aligned}
$$

Here $W_Q$, $W_K$, $W_V$ and $W_O$ all have shape `(hidden_size, hidden_size)`, and each $\text{head}_h$ has shape `(batch_size, seq_len, head_size)`.

Start with the `MultiHeadAttention` class. 
Read `MultiHeadAttention.forward` first. It projects the input to queries, keys and values, and uses `split_heads` to give each head its own slice. After attention, it uses `combine_heads` to concatenate the heads, and then applies the output projection. Make sure you understand how `split_heads` and `combine_heads` fit around the attention computation.

Then, complete the `scores`, `attention_weights` and `context` lines in `MultiHeadAttention.forward` with a standard implementation of scaled dot-product attention, as you did in your `attention` function. Note that `q`, `k` and `v` have shape `(batch_size, num_heads, seq_len, head_size)`, while `mask` has shape `(batch_size, seq_len, seq_len)`. Your code must handle the extra head dimension.

Next:
1. In `MultiHeadAttention.__init__`, set the sizes (hint: there is only one size that needs an actual computation). Then, define the three projections ($W_Q$, $W_K$, $W_V$) and the output projection ($W_O$). Replace each `...` with the value or layer it should hold, and keep the variable names.
2. In `MultiHeadAttention.split_heads`, reshape a tensor of shape `(batch_size, seq_len, hidden_size)` into `(batch_size, num_heads, seq_len, head_size)`, so that each head becomes its own "second batch" dimension. Read about `torch.Tensor.view`, `torch.Tensor.reshape` and `torch.Tensor.transpose`.
3. In `MultiHeadAttention.combine_heads`, apply the inverse operation: combine the activations of `(batch_size, num_heads, seq_len, head_size)` back into `(batch_size, seq_len, hidden_size)`. This is the "Concat" in the equation above.

The docstrings of `split_heads` and `combine_heads` include small examples. You can use these as extra test cases by making sure your implementation reproduces them **exactly**.

Then, test your implementation:

```
python test_1_3.py 5
```

Next, use your multi-head attention in a transformer block. You are given the initialization function of this class in `MultiHeadTransformerBlock.__init__`.
Implement `MultiHeadTransformerBlock.forward`, which is very similar to the `TransformerBlock` you implemented, with $\text{Attention}(Q, K, V, \text{mask})$ replaced by $\text{MultiHead}(X)$:

$$
\begin{aligned}
H &= \text{LayerNorm}_1(X + \text{MultiHead}(X)) \\
Y &= \text{LayerNorm}_2(H + \text{MLP}(H))
\end{aligned}
$$

Finally, you'll connect several such blocks to a single language model, in `MultiHeadTransformerLM`.
You are given the `init` function, and you should implement the `forward` function, similarly to the one from `MultiLayerTransformerLM`.

Test your implementation, and once it passes train the multi-head model:

```
python test_1_3.py 6 7
python lab-1-3-transformer-lm.py --model multihead --num-layers 2 --num-heads 4
```

How does the multi-head model compare to the multilayer model with the same number of layers?

### TODO 8: Why several heads?

**Question:** Which advantage is there for a multi-head attention mechanism compared to a single-head attention mechanism?

**Answer:**
Each head, having a different initialization also as a result learns something different.
This means each head captures a different underlying "connection" between token (grammar, location in sentence, semantic context, some other less well defined connection, ...)


### TODO 9: Getting the best model you can

Hyperparameter choice and tuning is often extremely important for high performance, and the transformer model you implemented has many such hyperparameters: model size (layers, heads per layer, embedding and hidden dimensions), batch size, amount of training data, number of epochs, and more.

Your goal in this part is to tune the hyperparameters to yield the best possible results: hypothesize which hyperparameter changes will yield positive changes to the results (validation loss, perplexity, and qualitative evaluation of the generated samples), change the parameters, and run the full script to check your hypotheses.

Note that some settings are exposed as command line flags, and the rest are fields of `Config`.

**Question:** List a few hyperparameter changes which you believe will yield better results. For each one, list the change, hypothesized effect, measured results, and explain the finding. 

**Answer:**
0. Base (2 layers, 4 heads default embedding and hidden dimensions)
valid loss 3.398, valid ppl 29.9; shuffled-target loss 9.366, shuffled-target ppl 11687.8

1. 4 layers instead of 2
More layers should result in better learning so we expect lower perplexity
We got the following results
valid loss 3.142, valid ppl 23.1; shuffled-target loss 9.716, shuffled-target ppl 16578.0
So like we expected we got lower ppl as a result of the added layers.

2. 8 head instead of 4
More heads should also lower the ppl but not as much as increasing the layers amount (since deep nets are exponentially better then flat net with the same parameter count, more heads are not so we expect a weaker result)
We got the following results
valid loss 3.376, valid ppl 29.3; shuffled-target loss 9.476, shuffled-target ppl 13042.3
So our expectation came to pass. Indeed the added head helped capture more meaning but the added meaning helped the results less than adding more layers

3. 4 layers instead of 2 & 8 head instead of 4
The combined improvement should be better than any one separately since each improves the results in "a different way"
We got the following results
valid loss 3.109, valid ppl 22.4; shuffled-target loss 9.823, shuffled-target ppl 18445.0
Once again our prediction came true (also the text we got was more coherent). The low valid ppl and high shuffled-target ppl are a good indication for the quality of the model

## Lab debrief
**Question:** We're interested in any thoughts you have about this lab so that we can improve this lab for later years, and to inform later labs for this year. Please list any issues that arose or comments you have to improve the lab. Useful things to comment on include the following, but you're not restricted to these:
* Was the lab too long or too short?
* Were the instructions clear?
* Was it clear (at least after you completed the lab) what the points of the exercises were?

**Answer:**
This lab took 2.5 hours, much more manageable than the last 2
Instructions were not always clear for the split head task, a hint about which function may be handy would have benn greatly appreciated
The point of the lab was clear
## Submission
Collect all of the files in the given directory into a **flat** zip file (**NO** internal directories) and submit it for grading via Gradescope.


## Further reading

Positional embeddings, absolute and relative:
- [Designing positional encoding](https://huggingface.co/blog/designing-positional-encoding)
- [Self-attention with relative position representations](https://arxiv.org/abs/1803.02155)
- [ALiBi](https://arxiv.org/abs/2108.12409)
- [RoPE (used in most modern transformer models)](https://www.youtube.com/watch?v=SMBkImDWOyQ)

Changes to the Transformer architecture since 2017:
- [What changed in the Transformer architecture](https://huggingface.co/blog/rishiraj/what-changed-in-the-transformer-architecture)
