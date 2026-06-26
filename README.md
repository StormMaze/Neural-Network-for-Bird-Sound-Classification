# Bird Sound Classification

A convolutional neural network that identifies bird species from audio recordings by converting bird calls into spectrogram images and classifying those images with a CNN trained from scratch.


## Overview

Rather than feeding raw audio into a model, this project turns the classification problem into an image problem i.e every `.mp3` recording is converted into a spectrogram (a visual representation of frequency content over time) and a CNN learns to recognize the visual "fingerprint" each species' call leaves behind. The dataset comes from Cornell Lab of Ornithology's [Birdcall Identification dataset](https://www.kaggle.com/c/birdsong-recognition), narrowed down to 6 species for this classifier.


## Results

Trained on 6 bird species, the model reached **90.5% accuracy** on a held out test set.

| Species code | Common name | Precision | Recall | F1 |
|---|---|---|---|---|
| `amecro` | American Crow | 1.00 | 1.00 | 1.00 |
| `bulori` | Baltimore Oriole | 0.90 | 0.90 | 0.90 |
| `chispa` | Chipping Sparrow | 0.82 | 0.82 | 0.82 |
| `grhowl` | Great Horned Owl | 0.91 | 0.91 | 0.91 |
| `horlar` | Horned Lark | 0.85 | 1.00 | 0.92 |
| `mouchi` | Mountain Chickadee | 1.00 | 0.80 | 0.89 |

**Overall: 90% accuracy, 0.91 macro F1** across 63 test samples. American Crow was a clean 100%. Chipping Sparrow was the model's weakest spot (0.82 F1), which tracks with how similar chipping sparrow calls can sound to other small songbird species in a spectrogram.


## Dataset

[Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition) (Kaggle / Cornell Lab of Ornithology) is a short labeled audio clips of bird calls, organized by [eBird species code](https://ebird.org/species/). Spectrograms were generated for 10 species, with 6 ultimately used for the trained classifier (100 images each, 600 images total)which are American Crow, Baltimore Oriole, Chipping Sparrow, Great Horned Owl, Horned Lark and Mountain Chickadee.


## Pipeline

1. **Audio → spectrogram** — each `.mp3` is loaded with `librosa`, converted to a 128 band spectrogram and rendered to a PNG image (no axes/labels just the raw visual pattern).

2. **Augmentation** — for the first 40 clips per species, a [SpecAugment](https://arxiv.org/abs/1904.08779) style augmentation is applied before saving.( random time masking and frequency masking blank out short stripes of the spectrogram, which helps the model generalize instead of memorizing exact call patterns.)

3. **Train/val/test split** — original (nonaugmented) images are stratified split 65/17.5/17.5 into train/val/test by species, then the augmented images are folded entirely into the training set. Final split: **474 train / 63 val / 63 test**.

4. **Image pipeline** — spectrogram PNGs are decoded, resized to 224×224 and normalized to `[-1, 1]`, batched at size 32 via `tf.data` (cached + prefetched for training speed).

5. **Model** i.e a CNN built from scratch:
- 5 convolutional blocks (32 → 64 → 128 → 256 → 256 filters) each followed by max pooling
- Dropout (0.2) after the conv stack then a flatten → Dense(256) → Dropout(0.5) → Dense(6) output head
- Compiled with Adam and sparse categorical cross entropy (logits, since there is no final softmax)

6. **Training** — up to 25 epochs with early stopping on validation loss (patience 3, best weights restored). A learning rate reduction callback is implemented but not currently wired into training.

7. **Evaluation** — accuracy/loss curves plotted across epochs then test set evaluation with a confusion matrix and a full per class precision/recall/F1 report.



## Tech Stack

- **Python**, **TensorFlow / Keras** (custom CNN, no pretrained backbone)
- **librosa** for audio loading and spectrogram generation
- **scikit-learn** for the stratified train/val/test split and classification report
- **seaborn / matplotlib** for the confusion matrix and training curves
- Built and run in **Google Colab**, with data stored on **Google Drive**


## Running It

This notebook expects the Cornell Birdcall Identification dataset unzipped on Google Drive, with each species' audio in its own folder:

```
ColabNotebooks/BirdSong/data/
├── train_audio/
│   ├── amecro/*.mp3
│   ├── bulori/*.mp3
│   └── ...
```

To run it:

1. Download the dataset from Kaggle and place `birdsong-recognition.zip` on your Drive (or point the notebook's unzip step at wherever you've stored it).

2. Open the notebook in Colab and mount your Drive when prompted.

3. Run the spectrogram generation cells first and this is the slow part, since every `.mp3` gets loaded and rendered to an image. Spectrograms are skipped if they already exist, so re running is safe.

4. Run the train/val/test split then the model training and evaluation cells.

To run locally, swap the Drive mount and paths for local directories and everything else (`librosa`, `tensorflow`, `scikit-learn`, `seaborn`) works the same outside Colab.


## Possible Next Steps

- Wire in the `ReduceLROnPlateau` callback that's already defined but currently commented out of training

- Expand to the remaining 4 species spectrograms already generated (Northern Waterthrush, Savannah Sparrow, Swainson's Thrush, Warbling Vireo) for a 10 class model

- Try a pretrained image backbone (e.g., MobileNetV2/EfficientNet on the spectrograms) and compare against the from scratch CNN

- Address the Chipping Sparrow confusion specifically as more training examples or finer spectogram resolution in that frequency range

- Experiment with classifying raw audio embeddings directly instead of spectrogram images


## Conclusion
Built end to end as an audio classification project with spectrogram generation, augmentation, the CNN architecture and evaluation tooling were all written from scratch using the Cornell Birdcall Identification dataset.

