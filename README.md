# Keras Convolutional Text Punctuator

Is a small experimental project to punctuate text using a embedding layer, single convolutional layer and output softmax layer written in Keras. In current state it attempt to locate any of ",;.!?" and place a dot in their location.

This project is being used in practice in my Android app Youtube Reader: [https://github.com/vackosar/youtube-reader](https://github.com/vackosar/youtube-reader).

## Inspirational Project

Punctuator 2 from Ottokar Tilk at: https://github.com/ottokart/punctuator2

> A bidirectional recurrent neural network model with attention mechanism for restoring missing inter-word punctuation in unsegmented text.
  
Project used framework not suitable for Android and harder to read. Furthermore used RNN instead of simpler CNN. 

## Data Preparation

First used EU Parliament transcription data, but that was expectantly suboptimal choice. Style of the speech was too far from average. The usual words were used and the sentence length seemed longer. I subjectively improved using News Commentary data.

After cleaning the data, word index is built from 20k most occurring words. Word index maps words to either their frequency number or to a default token in case they are not in 20k most common ones.

Number of samples used was within order of magnitude of millions.


## Model Overview

```python
    model.add(createEmbeddingLayer(wordIndex))
    model.add(Conv1D(512, 3, activation='relu'))
    if wordIndex is not None:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(LABELS_COUNT, activation='softmax'))
```

Input layer is provided with 30 word token integer numbers for which we predict if there was one character from ",;.!?". Tokens are aquired by mapping words using the word index described above. 

Embedding layer used is a projection of GloVe Embedding:
> GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 

Convolutional layer is single dimensional:
> shift invariant or space invariant artificial neural network (SIANN), based on its shared-weights architecture and translation invariance characteristics.[

Regularization preventing over-learning is performed by the Dropout Algorithm.

Last layer is simple two neuron fully or densely connected layer.
   
Prediction output consisting of two float values is computed as softmax of activations of the two output neurons.


### Full View
<img src="full-graph.png" width="800">

### Zoomed
<img src="partial-graph.png" width="800">

## Performance

- Precision: 92%
- Recall: 92%
- F Score: 92%

I am  not confident in Recall and F Score measures, but Precision one is accurate.

### Examples

Original Piece Of Test Data:
> their lives in Spanish and United States history classes. Vice President Joseph R. Biden Jr. discusses a myriad of foreign and domestic issues and Justice Elena Kagan breaks her silence. The book on Manning says teams must pressure him to beat him. Times Reader 2.0 Try it FREE for 2 full weeks. New discoveries in the Mush Valley in Ethiopia include 22 million year old fossils of mammals as well as evidence of leaf consumption by insects. And, Please, No Shirtless Shots. A Room for Debate forum on why the shootings in Arizona have not led to calls for more gun control. Letters Mementos or Mess?

Punctuated Piece Of Test Data:
> Their lives in spanish and united states history classes. Vice president joseph r. Biden jr. Jr. Discusses a myriad of foreign and domestic issues and justice elena kagan breaks her silence. The book on manning. Says teams must pressure him to beat him. Times reader 20. Try it free for 2 full weeks. New discoveries in the mush valley in ethiopia include 22 million year old fossils of mammals as well as evidence of leaf consumption by insects and please no shirtless shots. A room for debate forum on why the shootings in arizona have not led to calls for more gun control. Letters mementos or mess 

Punctuated Elon Musk Interview Youtube captions:
> Well first of all is it i think if somebody is doing something that is useful to the rest of society. I think that's a good thing like it doesn't have to change the world like you know if you make something that has high value to people. And frankly even if it's something if it's like just a little game or you know the some improvement in photo sharing or something. If it if it has a small amount of good for a large number of people that's i mean i think that's that's fine like stuff doesn't need to be change the world just to be good. But you know in terms of things that i think are most likely to affect the the future of humanity.