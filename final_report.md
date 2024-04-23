---
layout: default
title: Final Report
description: Present results throughout the semester
mathjax: true
---

## Final Report

### Introduction/Background
The intersection of music and machine learning has been a vibrant area of research, with significant breakthroughs in both music classification and generation. In 2002, Tzanetakis proposed three feature sets specifically for genre classification [1], which laid the groundwork for following studies. Li further refined feature extraction by proposing Daubechies Wavelet Coefficient Histograms [2]. Method-wise, Lerch [3] proposed a hierarchical system, granting expansion capabilities and flexibility. Model-wise, both deep-learning and traditional machine-learning approaches are applied and examined [4].

Our dataset comprises 10 genres, each with 100 audio files of 30 seconds in length, accompanied by image visual representations for each file. It also includes two CSV files: one details mean and variance of multiple features extracted from each 30-second song, and the other provides similar data, but the songs were split before into 3 seconds audio files.

For our project, we will be testing with the 30 seconds data, but training with two approaches, the 30 seconds ones and the 3 seconds ones.

We are mostly interested in tempo, root mean square (RMS), spectral features such as centroid, bandwidth, and rolloff capture different aspects of the frequency content, zero crossing rate measures the noisiness, MFCCs related to the timbral texture of the sound, harmony, chroma that captures the intensity of the 12 different pitch classes, etc. These are the more significant features in DL and traditional ML learning [4].

Link to the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

### Problem Definition
The primary issue addressed is the use of machine learning techniques to classify music into genres. The possibility of improving the existing music recommendation algorithms serves as its driving force. We can promote a deeper understanding of musical structures and music creation and give listeners a more engaging and personalized experience by appropriately classifying different genres of music. The issue is justified with economic concerns. When consumers are constantly presented with irrelevant suggestions, poor recommendation algorithms not only make the user experience worse but may also cause streaming platforms to incur financial losses. The project's goal is to strengthen the bond between listeners and music, which will benefit streaming services' bottom line as well as their audience's general contentment.

### Methods
Throughout the semester, we experimented with three strategies to solve this problem. Here are descriptions of each strategy. The discussions and reflections on each strategy is presented in a later section of this report:

#### Strategy 1: Stacked MFCC & Chroma
This idea was inspired by the notebook available on [kaggle](https://www.kaggle.com/code/eonuonga/gtzan-genre-classification-preprocessing-1-2).

In the data preprocessing phase, we extract a set of 12-channel Mel-Frequency Cepstral Coefficients (MFCC) and Chroma features, along with their first and second derivatives, from the raw audio signals. These features are commonly employed in music genre classification tasks due to their ability to effectively capture essential spectral and harmonic information present in the audio data. The complete data preprocessing workflow is shown in the following diagram:




### References
[1]: Tzanetakis, G. and Cook, P. (2002) 'Musical genre classification of Audio Signals', IEEE Transactions on Speech and Audio Processing, 10(5), pp. 293-302. doi:10.1109/tsa.2002.800560.\
[2]:  Li, T., Ogihara, M., & Li, Q. (2003, July). A comparative study on content-based music genre classification. In Proceedings of the 26th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 282-289).\
[3]: Burred, J. J., & Lerch, A. (2003, September). A hierarchical approach to automatic musical genre classification. In Proceedings of the 6th international conference on digital audio effects (pp. 8-11).\
[4]: Ndou, N., Ajoodha, R., & Jadhav, A. (2021, April). Music genre classification: A review of deep-learning and traditional machine-learning approaches. In 2021 IEEE International IOT, Electronics and Mechatronics Conference (pp. 1-6). IEEE\
[5] Bahuleyan, H. (2018). Music Genre Classification using Machine Learning Techniques. arXiv:1804.01149v1

### Contribution Table
![alt text]()

### Gantt Chart
Here is the [Gantt Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/ypan390_gatech_edu/EeUk8XSMMSFAqpbJ5cSKEDQBIkUN30qINQYGgmnCyVkJLg?e=4%3A6bQdYn&at=9&CID=8b4a2e17-0dca-5391-786c-d97bbece4005).

### Appendix
[Yuanming Luo's Proof](./CS4641ProjectIdeaAndTheory.pdf)


[back](./)
