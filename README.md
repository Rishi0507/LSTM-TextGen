# Next-Word Prediction with LSTM (PyTorch)

A from-scratch implementation of a word-level next-word prediction model using an LSTM in PyTorch, built to understand the full pipeline (tokenization, vocabulary building, sequence generation, padding, training, autoregressive generation) without relying on high-level NLP abstractions.

## Overview

The model is trained on a single small FAQ-style document and learns to predict the next word given a sequence of preceding words. After training, it generates text by repeatedly predicting and appending one word at a time.

## How It Works

1. **Tokenization**: The document is lowercased and tokenized with NLTK's `word_tokenize`.
2. **Vocabulary Building**: A word-to-index mapping is built from unique tokens, with an `<unk>` token for unseen words.
3. **Sequence Generation**: Each sentence is split into multiple training sequences via a sliding window (`[w1]`, `[w1,w2]`, `[w1,w2,w3]`, ...), where the last word is the prediction target.
4. **Padding**: All sequences are left-padded with zeros to the length of the longest sequence in the dataset.
5. **Model**: An embedding layer feeds an LSTM; the final hidden state is passed through a linear layer to produce a probability distribution over the vocabulary.
6. **Training**: Cross-entropy loss, Adam optimizer.
7. **Generation**: Greedy decoding: the model predicts one word, appends it to the input, and repeats.

## Model Architecture

```
Embedding(vocab_size, 100)
  → LSTM(100, 150, batch_first=True)
  → Linear(150, vocab_size)
```

## Dataset Stats

| Metric | Value |
|---|---|
| Source | Single FAQ document (~600 words) |
| Total lines / sentences | 137 |
| Training sequences generated | 631 |
| Vocabulary size | 294 |
| Max sequence length (after padding) | 17 |

## Why There's No Train/Validation Split

This is intentional, not an oversight, and it's worth being upfront about what that means for the results below.

- The entire dataset is **one document** turned into 631 short sequences over a 294-word vocabulary. That's small enough that a held-out split would be a handful of sequences, not a meaningful generalization test.
- The goal of this project was to implement the **mechanics** of a word-level language model end-to-end (vocab building, padding, batching, training loop, autoregressive generation), not to produce a model that generalizes to unseen text.
- As a direct consequence: **the loss curve below reflects fit to the training document, not generalization.** Low training loss here does not mean the model would perform well on a different FAQ document or any text outside this one. Any claims about model "performance" should be read as "performance on the data it was trained on," nothing more.
- If this is trained on a larger, more varied corpus in the future, adding a proper train/val split at that point would become meaningful.

## Visualizations

> Run the chart-generation code (kept separately, not in this README) after training, save the figures into an `assets/` folder, then they'll render below.

### Training Loss per Epoch
![Training Loss](assets/training_loss.png)

### Vocabulary Frequency Distribution
![Vocabulary Frequency](assets/vocab_frequency.png)

### Sequence Length Distribution
![Sequence Length Distribution](assets/sequence_length_dist.png)

### Word Embedding Space (PCA projection)
![Word Embeddings PCA](assets/embeddings_pca.png)

## Observations

- **Training loss**: An earlier version of this chart showed a misleadingly narrow range (roughly 4.12 down to 3.85), caused by re-running the training loop on an already-trained model instead of a freshly initialized one, so "epoch 1" on that chart was really resuming from a partially converged state. After resetting the model and optimizer before training, the loss now starts near the expected random-init baseline for this vocabulary size (ln(294), about 5.68) and declines from there across the 100 epochs.
- **Vocabulary frequency (words only)**: With punctuation filtered out, the most frequent words are dominated by stopwords ("the", "is", "you", "will", "can"), with the FAQ-specific word "program" also appearing frequently since it's one of the most repeated topic terms in the source document. This is expected for a small, single-document corpus.
- **Sequence length distribution**: The distribution is strongly skewed toward short sequences, with relatively few sequences near the maximum length of 17. This isn't a property of the source document's sentence lengths. It's a direct consequence of generating one training sequence at every prefix length for each sentence, so short prefixes get produced far more often than long ones by construction.
- **Word embeddings (PCA)**: There's no strong overall thematic clustering, consistent with the dataset being too small for embeddings to organize semantically in a clear way. That said, topic-relevant words such as "payment," "support," and "sessions" sit noticeably apart from the generic stopword cluster, a mild sign that the model differentiates topic words from filler words even with limited data.
- **Top-k prediction probabilities**: Not generated for this version of the project.

## Limitations

- **Greedy decoding**: the model always picks the highest-probability word, which tends to produce repetitive output on longer generations. Temperature or top-k/top-p sampling would likely produce more varied text.
- **No held-out evaluation**: as explained above, there's currently no way to measure generalization, only training-set fit.
- **Fixed max sequence length used at inference** is tied to this specific training document; swapping in a different corpus requires recomputing it.
- **Padding and unknown-word collision**: the padding value (0) is the same index used for `<unk>` in the vocabulary. During training this is harmless, since the vocabulary is built from the same document used for training, so there are no genuine out-of-vocabulary tokens. At generation time, however, a genuinely unseen word would map to the same index as padding, so the model can't distinguish a real unknown word from empty padding.
