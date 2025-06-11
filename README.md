# IMDB_Classification_LSTM_CNN
## Project Overview
This project explores a multimodal approach to genre classification of movies using both image (poster) and text (overview) data from the IMDB dataset. Two deep learning models are implemented:

A Convolutional Neural Network (CNN) to classify movies based on their posters.

A Long Short-Term Memory network (LSTM) to classify movies based on their textual overviews.

Both models are built for a multilabel classification task, as each movie can belong to multiple genres.

## Models
### CNN (Poster-based)
21-layer deep CNN for image feature extraction.

Layers include convolution, pooling, dropout, and fully connected classification layers.

Sigmoid activation used for multilabel output.

Designed to handle standardized and batched poster images.

### LSTM (Overview-based)
Uses a Recurrent Neural Network (RNN) with LSTM cells to handle textual sequences.

Addresses the vanishing gradient problem common in RNNs.

Tokenizes overviews with a vocabulary of 10,000 words.

Ends with sigmoid activation for multilabel output.

### Preprocessing
CNN
Posters split into training (80%) and validation (20%) sets.

Standardized in size and vectorized.

Batching (64 samples), caching, and prefetching for training efficiency.

LSTM
Overviews split similarly (80/20).

Text vectorized and encoded using a vocabulary tokenizer.

Preprocessed into batched datasets for training.

### Training Configuration
Optimizer: Adam

Loss Function: Binary Crossentropy (for multilabel classification)

Evaluation Metrics: Precision & Recall

Callbacks: ModelCheckpoint for saving best models based on validation loss.
