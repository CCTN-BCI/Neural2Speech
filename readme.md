# The codes and speech waveforms

Here is the [speech demo link](https://cctn-bci.github.io/Neural2Speech/).

## Directory: speech_waveform

- `raw_speech`: Raw speech files.
- `speech_autoencoder`: Speech files reconstructed using an autoencoder.
- `mlp`: Speech files processed using a multilayer perceptron (MLP).
- `unlockGAN`: Speech files processed using the LSTM-CNN regression approach.
- `Neural2Speech`: Results from our proposed framework.

## Directory: codes

- `train_SAE.py`: Code for training the speech autoencoder (training phase 1).
- `Dataset_fin_multi.py`: Code for dataset implementation.
- `model_lstm_and_SAE.py`: Code for model implementation.
- `reconstruction.py`: Code for adaptor training (training phase 2)