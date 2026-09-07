# sentencenet

A staged implementation of a word-level language model, similar to Andrej Karpathy's character-level model, as in [makemore](https://github.com/karpathy/makemore).

I start with a simple MLP and gradually change the various components of the model, including the optimizer, layer types, and learning rate.

The model is trained on the Penn Treebank dataset<sup>1</sup>, downloaded from [Hugging Face](https://huggingface.co/datasets/FALcon6/ptb_text_only).

The goal of this project is to study how well small language models can produce human-readable text. I achieved various levels of success throughout these experiments.

## Abstract

I start with a simple MLP, using stochastic gradient descent and simple linear layers, along with layer normalization and `tanh` nonlinearity. Each example consists of 8 tokens, with each token being one word or one part of a word (in the case of `'s` and others). This MLP predicts the individual probabilities of the next tokens.  

Piece by piece, I assemble more complex (and thus, more powerful) models, analyzing their performance along the way through measuring cross-entropy loss on training and validation sets, as well as sampling sentences from the model. The PyTorch API is used to quickly iterate on model architectures. However, the implementation is kept highly custom to make experimentation easier, using custom layers when needed.

I use the Penn Treebank dataset in this project, which contains a 10,000-token vocabulary, which replaces words outside of the selected vocabulary with the `<unk>`. Numbers are also replaced by the `N` token.

## Repository structure

| Name | Role |
| ---- | ---- |
| `data/` | Directory with train, validation, and test sets |
| `mlp.ipynb` | Word-embedding MLP; contains the first two experiments |
| `rnn.ipynb` | Word-embedding RNN; contains the last two experiments |
| `tracking.md` | Notes from the implementation process |

## Methods

The model's vocabulary contains 10,000 individual words and word parts. Each is assigned an index from `0` to `9999`.

From the training set, I create `887521` training examples by sliding an eight character context length over each sentence in the dataset.

### Split

| Split | # of Sentences | Approximate Percentage (%) |
| ----- | --------- | ------- |
| `train` | `42068` | `85.5` | 
| `validation` | `3370` | `7.0` | 
| `test` | `3761` | `7.5` |

### Architecture

| Version | Description | # of parameters | Optimizer |
| --- | --- | --- | --- |
| v1 | MLP with word embedding, linear layers, layer normalization<sup>3</sup> tanh nonlinearity, and WaveNet-style<sup>4</sup> flattening (see note #3) | `x` | Stochastic gradient descent (SGD) |
| v2 | Same architecture as above  | `x` | AdamW |
| v3 | RNN with word embedding, four recurrent layers<sup>5</sup>, and tanh nonlinearity | `x` | AdamW |
| v4 | Same architecture as above, but with layer normalization and WaveNet-style flattening | `x` | AdamW |

#### Notes
1. Embedding dimensionality is kept constant, at `32`
2. Learning rate was adjusted slightly when switching to the AdamW optimizer (`0.01` to `0.01` or `0.025`), which typically works better when it is set 
3. Flattening of input sequence is inspired by the Google DeepMind's WaveNet

### v1 - MLP with Stochastic Gradient Descent
#### Architecture
```
previous 8 words -> 32-dim embedding -> linear + layernorm + tanh + flatten (4x) -> next word probabilities
```

#### Description

I build a simple word-level MLP that takes the previous eight words as input, embeds them in a 32-dimensional space, and then processes the context through a stack of WaveNet-inspired blocks. 

Each block applies a linear projection, layer normalization, and tanh nonlinearity before flattening the sequence. The sequence is flattened into a single representation by the time it passes through every block. 

The final linear layer maps the hidden state to logits over the vocabulary, producing a probability distribution for the next word.

This is the baseline model for the project: deliberately small and easy to interpret. It establishes a reference point for the later experiments. At this stage, the model learns basic word-to-word transitions, but it remains limited by the lack of explicit temporal structure beyond the fixed 8-token context.

### v2 - MLP with AdamW
#### Architecture
```
previous 8 words -> 32-dim embedding -> linear + layernorm + tanh + flatten (4x) -> next word probabilities
```

#### Description

I keep the same general architecture as v1, but replace stochastic gradient descent with the AdamW optimizer.

The goal of this version is to test whether a more modern optimizer improves the model's training without changing the underlying architecture. AdamW helps the model optimize more smoothly than plain SGD, leading to lower training and validation loss while maintaining the same recurrent-style MLP structure. 

At this point, the sampled text begins to become noticeably more readable and coherent, though some overfitting starts to occur.

### v3 - RNN
#### Architecture
```
previous 8 words -> 32-dim embedding -> 4-layer RNN w/ tanh -> WaveNet-style flattening -> next word probabilities
```

#### Description

I replace the fixed MLP blocks from the earlier experiments with a recurrent stack. Each token in the previous eight-word context is embedded in a 32-dimensional space and passed through a 4-layer RNN with tanh nonlinearity. The recurrent hidden states are then flattened and projected to the vocabulary, producing probabilities for the next word.

The goal of this version is to test whether recurrence across the context window is more useful than the strictly feedforward MLP. 

The RNN is substantially more expressive, and it reaches lower training loss than the earlier models. However, it also overfits much more aggressively, with the validation loss increasing sharply relative to the training loss, suggesting that the model is learning highly specific sequence patterns rather than generalizing well.

### v4 - RNN + Layer Normalization
#### Architecture
```
previous 8 words -> 32-dim embedding -> 4-layer RNN w/ layer normalization in between each recurrent layer -> WaveNet-style flattening -> next word probabilities
```

#### Description

I extend the recurrent model from v3 by applying layer normalization to the hidden states of the recurrent stack before the final projection. The main goal is to reduce the large training/validation gap in the unnormalized RNN. 

As in the MLP experiments, I keep a WaveNet-inspired flattening strategy so the 8-token context is gradually reduced to a single representation before the final linear layer produces the next-word probabilities.

## Results

Each model is evaluated on cross-entropy loss. Throughout the experimentation process, I occasionally sample from the model to inspect the quality of the sentences.

### Cross-Entropy Loss

I am currently in the process of evaluating the model's performance on the test set. These will be added in the near future.

| Version | Training Loss | Validation Loss | Notes | 
| --- | --- | --- | --- |
| `v1` | `5.684` | `5.702` | Baseline performance. All following models outperformed it on training loss. All but `v3` beat outperformed it on validation loss. |
| `v2` | `4.836`  | `5.253` | At this point, sentences sampled from the model start to become somewhat readable (see "Sampling"). However, slight overfitting is taking place. |
| `v3` | `4.047` | `5.911` | Severe overfitting. RNNs are significantly more powerful, so that is understandable |
| `v4` | `5.190` | `5.469` | Overfitting isn't as bad, but samples from the model are significantly worse than those from `v2` (see "Sampling"). | 

### Sampling

Sentences sampled from `v2`:

```
but then no sign we are no step allowed the second flow of my sources 

there 's recently are opposed mr. <unk> said <unk> does n't generates <unk> seven 

the <unk> familiar with daiwa laboratory is tested as regarded as lead of plaintiffs from american times while inadequate stock and 

the tokyo report fell N points in an elderly 

the charges includes N N east partnership has lost N N to N N N senior subordinated one-year buy-out preferred transport debt via <unk> at prudential-bache 
```

Sentences sampled from `v4`:

```
he says this 

for like 

the closed-end herbert <unk> rain who write to personally dr. 

chairman a december days N 

your tends 

de predictable barber 
```

The probability of the stop token (`<n>`) seems to have been significantly increased during the training of `v4`, as compared to previous versions (more `<n>` tokens -> shorter sentences)

## Setup

This project is built with Python and PyTorch. The notebooks assume that the dataset parquet files are available in the project root.

1. Clone the repository and move into it:
   ```powershell
   git clone https://github.com/<your-user>/sentencenet.git
   cd sentencenet
   ```

2. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If you want the CPU-only PyTorch build, this is usually sufficient. If you have a CUDA-capable GPU, you may prefer the CUDA-enabled build for better training speed.

4. Ensure the dataset files are present in the repository root:
   - `train.parquet`
   - `validation.parquet`
   - `test.parquet`

5. Launch the notebooks:
   ```powershell
   jupyter lab
   ```
   or
   ```powershell
   jupyter notebook
   ```

6. Open either `mlp.ipynb` or `rnn.ipynb` and run the cells in order. The notebooks initialize the vocabulary and dataset from the parquet files automatically.

Notes:
- The project uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, so it will run on CPU or GPU depending on your machine.
- If the notebook fails to find the dataset, confirm that the parquet files are in the expected folder and update the paths in the notebook accordingly.