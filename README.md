# Women's Clothing Reviews – Attention-Based Sentiment Model
## Project Overview

This project analyzes customer reviews from an e-commerce clothing dataset using a custom attention-based neural network inspired by the Transformer architecture.

The model aims to learn meaningful representations from textual reviews through:

Word embeddings

Scaled dot-product attention

Feed-forward layers

Residual connections

## Dataset

### Dataset: Women's Clothing E-Commerce Reviews
### Source: Kaggle

The dataset contains real customer reviews along with structured metadata.

Main features include:

Review Text

Rating

Recommended Indicator

Clothing ID

Age

Division Name

Department Name

Class Name

## Data Preprocessing

Several preprocessing steps were applied before training:

Missing review texts filled with empty strings

Text tokenization using Keras Tokenizer

Vocabulary size limitation

Sequence padding

Example:

df["Review Text"] = df["Review Text"].fillna("")
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=max_len)

### Model Architecture

The model implements a simplified Transformer-style pipeline.

### Embedding Layer

Transforms tokenized words into dense vector representations.

### Attention Mechanism

Scaled dot-product attention is used to compute contextual relationships between tokens:

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

Components:

Query (Q)

Key (K)

Value (V)

### Feed-Forward Network

Two dense layers are applied to the attention output, followed by a residual connection.

### Pooling Layer

Global Average Pooling is used to reduce sequence dimensions.

### Output Layer

A sigmoid classifier produces the final prediction.

## Technologies Used

Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib

## Visualization

Recommendation distributions were analyzed across clothing classes.

The analysis includes:

Recommended vs Not Recommended counts

Class-based breakdowns

## Key Concepts Practiced

This project focuses on strengthening understanding of:

Embedding layers

Attention mechanisms

Matrix operations

Residual connections

Sequence modeling

## Notes and Limitations

This implementation is designed primarily as a learning and experimentation project rather than a production-ready model.

Simplifications include:

Single-head attention

Lightweight embedding size

Basic classification head

Potential future improvements:

Multi-head attention

Advanced evaluation metrics

Hyperparameter tuning

Improved label engineering

## Purpose of the Project

The project was developed to:

Deepen understanding of Transformer mechanics

Practice custom neural network design

Explore NLP workflows

## Author

Developed as part of deep learning and NLP practice.
