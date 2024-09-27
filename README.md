# TTS Quality assessment

This repo  is about quality assessment of a recent tts engine `jenny` that is fine-tuned version of `parler-tts`

## Setup environment

`conda create -n jenny python==3.8`

`pip install -r requirements.txt`

## Data

The datasets of interest are: 

1. Librispeech (test-clean)
    - Diverse content from audiobooks with clean audios
2. Librispeech (test-other)
    - Diverse content from audiobooks with noisy audios
3. Jenny
    - Extracted 1000 wavs from training set for testing SIM-o purposes

The audio lengths are distributed from  4 to 10 seconds, following Voicebox paper.

The LibriSpeech dataset can help us to understand model correctness and intelligibility on both clean / noisy scenarios
The Jenny dataset is used to compare generated speech to the original speech.

## Metrics

The speech can be decomposed into several attributes such as Intelligibility, Timbre and Prosody. For the TTS task the model should also generate diverse and realistic samples

Following the Voicebox paper we calculate assessment metrics accroding to: 

1. Correctness and intelligibility
    - Word Error Rate (WER)
    - Character Error Rate (CER)
    - Calculating intelligibility according to WER and CER of transcriptions from ASR model is widely adopted in the literature.
2. Coherence (Timbre)
    - Similarity against the original audio (SIM-o)
    - The cosine similarity between speaker verification embeddings from generated and real samples is a good measure of voice similarity 

3. Prosody:
    - Mel Cepstral Distortion with Dynamic Time Warping (MCD)
    - The MCD metric compares k-th (default k=13) Mel Frequency Cepstral Coefficient (MFCC) vectors derived from the generated speech and ground truth, respectively.

4. Diversity and quality
    - Fréchet Speech Distance (FSD) 
    - FSD captures the similarity between generated and real audios at the distribution level in  wav2vec 2.0 feature space. It is widely adopted in image generation evaluations



## Text descriptions

`Default`: Jenny speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality. 

`Enthusiastic`: Jenny speaks at an above average pace with an enthusiastic tone in a very confined sounding environment with clear audio quality.

`Dramatic`: Jenny speaks at an average pace with an dramatic tone in a very confined sounding environment with clear audio quality.

The idea of proposed `Enthusiastic` and  `Dramatic` descriptions is to assess tts performance given emotional speech. Will it even generate emotional speech?

## Results
For the following sections the text descriptions is `Default`
### Dataset `jenny`
|               | WER      |   CER    | SIM-o  | FSD     |   MCD    |
| :-----------: | :------: | :-------:| :----: |   :---: | :---:    |
| ground truth  |  0.1333  |   0.0430 | 0.9698 |  0.0515 |    0.00  |
| jenny TTS     |  0.0000  |   0.0000 | 0.9566 |  0.1858 |   14.71  |

The `jenny TTS` models intelligibility on training set is better than the intelligibility of the original speaker. Additionally, I've listend to couple of wavs, the original speaker misspells words sometimes.

 The   `ground truth` SIM-o is measured as mean similarity between halfs of speech recordings. The `jenny TTS` has high SIM-o score, just below the `ground truth` score, so the similarity to the original speaker is high

 However, there is still a gap between real distribution and synthesized distribution accroding to FSD, and the MCD score is quite high


### Dataset `librispeech-test-clean`

|               | WER      |   CER    | SIM-o  | FSD     |   MCD    |
| :-----------: | :------: | :-------:| :----: |   :---: | :---:    |
| ground truth  |  0.0556  |   0.0233 | 0.9478 |  0.0391 |    0.00  |
| jenny TTS     |  0.1111  |   0.0698 | 0.4260 |  4.6197 |   16.79  |

We see that `jenny TTS` has twice as much WER as `ground truth`, at around 11%. So, the intelligibility definetly has a room to imporve.  The SIM-o score is quite low at 0.4260, also the FSD is quite high, which is totally normal, since  `jenny TTS` is the single speaker tts. The prosody score for `librispeech-test-clean` is lower than for `jenny` dataset, since the model tries to mimic prosody of `jenny`.


### Dataset `librispeech-test-other`

|               | WER      |   CER    | SIM-o  | FSD     |   MCD    |
| :-----------: | :------: | :-------:| :----: |   :---: | :---:    |
| ground truth  |  0.0455  |   0.0092 | 0.9745 |  0.0334 |    0.00  |
| jenny TTS     |  0.5     |   0.3853 | 0.9194 |  2.2128 |   15.59  |

The other subset consists of more challenging texts for model to synthesize. The intelligibility degrades drastically, the error occurs in 50% of words. The similarity scores for such generation are unreasonably high. I think that the speaker verification model is not robust for such tts outputs. The FSD and MCD score on `librispeech-test-other` dataset are still lower compared to scores on `jenny` dataset.

### Descriptions

The scores are computed on `librispeech-test-clean` dataset

|                               | WER      |   CER    | SIM-o  | FSD     |   MCD    |
| :-----------:                 | :------: | :-------:| :----: |   :---: | :---:    |
| ground truth                  |  0.0556  |   0.0233 | 0.9478 |  0.0391 |    0.00  |
| jenny TTS                     |  0.1111  |   0.0698 | 0.4260 |  4.6197 |   16.79  |
| jenny TTS + `Enthusiastic`    |  0.1667  |   0.1395 | 0.4546 |  0.938  |   17.76  |
| jenny TTS  + `Dramatic`       |  0.1111  |   0.0698 | 0.4024 |  4.911  |   16.98  |


Subjectivaely, I don't hear a lot of change in emotion. Overal, the scores indicate that the models with `Enthusiastic` and `Dramatic` descriptions perform worse. The exception is FSD score that is better for some reason 

## Conclusion 

Overall, the `jenny TTS` model copies the real `jenny` voice resonably well according to the SIM-o scores. The quality and diversity of generated samples can be improved. Also, it struggles on out of training texts, the itelligibility in such scenarios can be up to 50% WER on hard cases. The MCD scores indicate that generated prosody is not close to the real prosody. The model ability to change emotion based on text description seems low, but maybe the further investigation in other text descriptions is needed. 
