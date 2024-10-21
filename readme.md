# Natural speech re-synthesis from direct cortical recordings using pre-trained encoder-decoder framework
## Authors
Jiawei Li, Chunxu Guo, Li Fu, Lu Fan, Edward F. Chang, Yuanning Li

[Neural2Speech: A Transfer Learning Framework for Neural-Driven Speech Reconstruction](https://ieeexplore.ieee.org/document/10446614)
[Natural speech re-synthesis from direct cortical recordings using pre-trained encoder-decoder framework (to be uploaded into BioArxiv)]

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

## Temporal landmark detection and automatic phoneme recognition
Temporal landmark detection and automatic phoneme recognition can be effectively implemented using Hugging Face's tools and libraries. You can find more information and resources on their official website: [Hugging Face](https://huggingface.co).
In our study, onset detectors act as binary classifiers that identify the presence or absence of phoneme, syllable, or word onsets in speech. These classifiers analyze fixed-length segments of speech waveforms, using 0.04 seconds for phonemes and 0.2 seconds for syllables and words. The output is a scalar value between 0 and 1, indicating the probability of a temporal landmark's existence.
To tackle the challenge of imbalanced datasets, we applied data augmentation techniques during training, such as volume adjustments, speed alterations, pitch shifts, and the addition of Gaussian noise. Both positive and negative speech segments were sampled equally at 5000 instances for the onset detector. This approach helped to reduce potential biases arising from the uneven distribution of samples.
For quantitative evaluation of synthesized speech, we trained a phoneme recognizer that utilizes Wav2Vec2.0 as a feature extractor, along with a logistic regression classifier and a phoneme decoder. Similar to the onset detection process, the classifier generates a probability vector based on Wav2Vec2.0 features. The phoneme decoder then translates these probability vectors into a phoneme sequence, allowing us to accurately recognize phonemes.

## Copyright
To clarify ownership of this work, it is hereby stated that the copyright belongs to Jiawei Li. This work is supported by NSFC General Program 32371154 and Shanghai Pujiang Program 22PJ1410500. Correspondence to: Yuanning Li (liyn2@shanghaitech.edu.cn).
