# Neural2Speech: A Transfer Learning Framework for Neural-Driven Speech Reconstruction
## Authors
Jiawei Li, Chunxu Guo, Li Fu, Lu Fan, Edward F. Chang, Yuanning Li
## Abstract
Reconstructing natural speech from neural activity is vital for enabling direct communication via brain-computer interfaces. Previous efforts have explored the conversion of neural recordings into speech using complex deep neural network (DNN) models trained on extensive neural recording data, which is resource-intensive under regular clinical constraints. However, achieving satisfactory performance in reconstructing speech from limited-scale neural recordings has been challenging, mainly due to the complexity of speech representations and the neural data constraints. To overcome these challenges, we propose a novel transfer learning framework for neural-driven speech reconstruction, called Neural2Speech, which consists of two distinct training phases. First, a speech autoencoder is pre-trained on readily available speech corpora to decode speech waveforms from the encoded speech representations. Second, a lightweight adaptor is trained on the small-scale neural recordings to align the neural activity and the speech representation for decoding. Remarkably, our proposed Neural2Speech demonstrates the feasibility of neural-driven speech reconstruction even with only 20 minutes of intracranial data, which significantly outperforms existing baseline methods in terms of speech fidelity and intelligibility.

## A demo page of speech waveforms
This webpage showcases the results of a research project investigating neural activity reconstruction for speech. It features five sets of WAV files, each containing five recordings of the same sentence, including the original speech, re-synthesized by an autoencoder, and reconstructions produced in various ways from neural activity.
Here is the [speech demo link](https://cctn-bci.github.io/Neural2Speech/).

## The codes and speech waveforms
### Directory: speech_waveform

- `raw_speech`: Raw speech files.
- `speech_autoencoder`: Speech files re-synthesized using an autoencoder.
- `mlp`: Speech files processed using a multilayer perceptron (MLP).
- `unlockGAN`: Speech files processed using the LSTM-CNN regression approach.
- `Neural2Speech`: Results from our proposed framework.

### Directory: codes

- `train_SAE.py`: Code for training the speech autoencoder (training phase 1).
- `Dataset_fin_multi.py`: Code for dataset implementation.
- `model_lstm_and_SAE.py`: Code for model implementation.
- `reconstruction.py`: Code for adaptor training (training phase 2)

## Copyright
To clarify ownership of this work, it is hereby stated that the copyright belongs to Jiawei Li. This work is supported by NSFC General Program 32371154 and Shanghai Pujiang Program 22PJ1410500. Correspondence to: Yuanning Li (liyn2@shanghaitech.edu.cn).
