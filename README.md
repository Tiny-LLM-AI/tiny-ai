# Tiny Arithmetic Transformer

A GPT-style Transformer with 2,847 learnable numbers that learns single-digit arithmetic (`2+3=5`, `1-3=-2`, `6/2=3`), together with a web UI that shows **every matrix, every gradient and every weight update**.

The goal is understanding, not performance. The model is written in plain TypeScript with no machine-learning library: the forward pass, the backward pass (backpropagation) and the optimizers are all hand-written and short enough to read.

## What you can see in the web UI

1. **Data & model**: the vocabulary, how one example becomes several next-character training samples, and the list of all learnable matrices.
2. **Ask: forward pass**: type a question and follow the computation stage by stage: token ids, embedding lookup, position embedding, Q/K/V, attention scores with the causal mask, softmax, weighted values, residual connections, layer norm, the feed-forward network, logits and the final probabilities. Click any cell of any result matrix to see the exact multiply-and-add that produced it.
3. **Train: backward pass**: train for many epochs and watch loss and accuracy, or train on a single sample and inspect the loss, the error signal at the output, how much gradient each parameter receives, and each weight before and after the update (with the exact SGD or Adam arithmetic per weight).

## Getting started

Requires Node.js 22 or newer.

```bash
npm install
npm run dev        # web UI on http://localhost:5173
```

Command line:

```bash
npm run train              # trains and writes tiny-model.json (about 10 seconds)
npm run ask -- "2+3="      # loads tiny-model.json and answers step by step
npm test                   # checks backpropagation against numerical gradients
```

## How the model works

```text
"2+3="                          characters
  → [0,0,0,0,8,3,9,14]          token ids, padded on the left to 8 positions
  → tokenEmbedding rows         8 × 16 matrix, one vector per position
  + positionEmbedding           so the model knows where each character is
  → Transformer block
      Q, K, V = X · W + b       three learned projections
      softmax(Q·Kᵀ/√d + mask)   attention weights, the mask hides future positions
      · V                       mix information between positions
      residual + layer norm
      ReLU MLP                  process each position on its own
      residual + layer norm
  → last position · outputWeight → 15 logits, one per vocabulary character
  → softmax                      probabilities
  → pick the most likely         "5", then repeat until END
```

Training uses cross-entropy loss on the next character, backpropagation to compute the gradient of every weight, and Adam (or plain SGD) to update the weights. The model never sees a rule such as "if the operator is + then add": everything it knows is stored in its weight matrices.

## Project structure

```text
packages/
  core/                         The model, no dependencies
    src/
      config.ts                 Model size and training settings
      math/matrix.ts            Matrix type and operations (matMul, softmax, ...)
      math/random.ts            Seeded random numbers, so every run is reproducible
      data/dataset.ts           Loads and validates the arithmetic dataset
      data/tokenizer.ts         Character ↔ token id
      data/training-samples.ts  Example → next-character training samples
      model/parameters.ts       All learnable matrices and their initialization
      model/forward.ts          Forward pass, records every intermediate matrix
      model/layer-norm.ts       Layer norm forward and backward
      model/backward.ts         Backpropagation
      training/optimizer.ts     SGD and Adam
      training/trainer.ts       Batches, epochs, single-step reports
      inference/generate.ts     Greedy next-character generation
      persistence/model-file.ts Save and load tiny-model.json
      cli/                      train and ask commands
    test/gradient-check.test.ts Verifies backpropagation numerically
  web/                          React + Vite UI
    src/components/             Matrix heatmap, dot-product and softmax explainers, charts
    src/views/                  Data, forward pass and training views
train-data/arithmetic.json      60 single-digit examples
```

## Changing the dataset

Edit `train-data/arithmetic.json`. Each row is `{ "question": "2+3=", "answer": "5" }`. Rows with a wrong answer are rejected when the dataset is loaded.

## What this project does not cover

This is only the core mechanism: a Transformer learning to predict the next character from supervised examples. Chat models like ChatGPT add three more stages on top of the same mechanism, at a vastly larger scale: pretraining on huge amounts of text, instruction tuning on question/answer conversations, and reinforcement learning from human preferences.
