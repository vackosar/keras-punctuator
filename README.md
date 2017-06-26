# Keras Convolutional Text Punctuator

Is a small experimental project to punctuate text using a embedding layer, single convolutional layer and output softmax layer written in Keras. In current state it attempt to locate any of ",;.!?" and place a dot in their location.

This project is being used in practice in my Android app Youtube Reader [https://github.com/vackosar/youtube-reader](https://github.com/vackosar/youtube-reader).


## Performance


## Data Preparation



## Model Overview

```python
    model.add(createEmbeddingLayer(wordIndex))
    model.add(Conv1D(512, 3, activation='relu'))
    if wordIndex is not None:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(LABELS_COUNT, activation='softmax'))
```

Input layer is provided with 30 words for which we predict if there was one character from ",;.!?".  

Embedding layer used is a projection of GloVe Embedding:
> GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 

Convolutional layer is single dimensional:
> shift invariant or space invariant artificial neural network (SIANN), based on its shared-weights architecture and translation invariance characteristics.[

Regularization preventing overlearning is performed by Dropout algorithm.

Last layer is simple two neuron fully or densely connected layer.
   
Prediction output consisting of two float values is computed as softmax of activations of the two output neurons.


### Full View
<img src="full-graph.png" width="512" height="250">

### Zoomed
<img src="partial-graph.png" width="512" height="250">