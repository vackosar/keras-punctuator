'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
'''

from __future__ import print_function

import os
import re

import numpy as np

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys

BASE_DIR = 'D:\\IdeaProjects\\data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
WORDS_PER_SAMPLE_SIZE = 20
LABELS_COUNT = 21
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
SAVE_SAMPLED = False

# cat europarl-v7.en.clean.txt |grep -o '[,\.!?]'|wc -l
# 5102642
# cat europarl-v7.en.clean.txt |wc -w
# 53603063
# 53603063 / 5102642 = 10.5
DOT_WORD = ' ddoott '
WORDS_PER_PUNCT = 11

# How large vocab? http://conversationsdirect.com/index.php?option=com_content&view=article&id=142%3Ahow-many-words-do-you-need-to-know-to-understand-english&catid=68%3Aarticles&Itemid=149&lang=en
# 2^13 = 8192
VOCAB_SIZE = 8192


# Clean and label data

def cleanData():
    toDelete = re.compile(
        '.*Resumption of the session.*|.*VOTE.*|^Agenda$|.*report[ ]*$|^$|^\.$|([^)]*)|[^a-z0-9A-Z\',\.?! ]')
    with open(BASE_DIR + "/europarl-v7/europarl-v7.en.clean.txt", 'w', encoding="utf8") as output:
        with open(BASE_DIR + "/europarl-v7/europarl-v7.en", encoding="utf8") as input:
            for fullLine in input:
                line = fullLine.rstrip()
                toDelete.sub('', line)
                if len(line) == 0:
                    continue
                output.write(line + " ")


def sampleData():
    import itertools

    def readwords(mfile):
        byte_stream = itertools.groupby(
            itertools.takewhile(lambda c: bool(c),
                                map(mfile.read,
                                    itertools.repeat(1))), str.isspace)

        return ("".join(group) for pred, group in byte_stream if not pred)

    def isMovingWindow(step):
        return step % 3 != 0

    NO_DOT_LIKE_LABEL = WORDS_PER_SAMPLE_SIZE
    SAMPLE_COUNT = 200000
    samples = []
    labels = []
    with open(BASE_DIR + "/europarl-v7/europarl-v7.en.samples.txt", 'w', encoding="utf8") as output:
        with open(BASE_DIR + "/europarl-v7/europarl-v7.en.clean.txt", 'r', encoding="utf8") as input:
            window = []
            step = 0
            dotLike = re.compile('.*[,\.?!]')
            for word in readwords(input):
                if len(window) < WORDS_PER_SAMPLE_SIZE:
                    window.append(word)
                    continue
                if step > SAMPLE_COUNT:
                    break
                step += 1
                if isMovingWindow(step):
                    window.append(word)
                    window.pop(0)
                    continue
                label = NO_DOT_LIKE_LABEL
                for index, queued in enumerate(window):
                    if dotLike.match(queued) is not None:
                        label = index
                sample = ' '.join(window)
                samples.append(sample)
                labels.append(label)
                if SAVE_SAMPLED:
                    output.write(sample)
                    output.write(' ' + str(label))
                    output.write('\n')
    return labels, samples


def tokenize(labels, samples):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(samples)
    tokenized_samples = tokenizer.texts_to_sequences(samples)
    padded_samples = pad_sequences(tokenized_samples, maxlen=WORDS_PER_SAMPLE_SIZE)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    tokenized_labels = to_categorical(np.asarray(labels))

    print('Shape of padded_samples tensor:', padded_samples.shape)
    print('Shape of tokenized_labels tensor:', tokenized_labels.shape)

    return tokenized_labels, padded_samples, word_index


# split the data into a training set and a validation set
def splitTrainingAndValidation(labels, data):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train, y_train, x_val, y_val


def indexEmbeddingWordVectors():
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def prepareEmbeddingMatrix(word_index, embeddings_index, nb_words):
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
def createEmbeddingLayer(nb_words, embedding_matrix):
    return Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=WORDS_PER_SAMPLE_SIZE,
                                trainable=False)


def trainModel(embedding_layer, x_train, y_train, x_val, y_val):
    print('Training model.')
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(WORDS_PER_SAMPLE_SIZE,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(LABELS_COUNT, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # happy learning!
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=10, batch_size=128)


def main():
    # cleanData()
    sampleData()
    labels, samples = sampleData()
    tokenized_labels, tokenized_samples, word_index = tokenize(labels, samples)
    x_train, y_train, x_val, y_val = splitTrainingAndValidation(tokenized_labels, tokenized_samples)
    embeddings_index = indexEmbeddingWordVectors()
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = prepareEmbeddingMatrix(word_index, embeddings_index, nb_words)
    embedding_layer = createEmbeddingLayer(nb_words, embedding_matrix)
    trainModel(embedding_layer, x_train, y_train, x_val, y_val)


main()
