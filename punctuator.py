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
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential
import sys

BASE_DIR = 'D:\\IdeaProjects\\data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
WORDS_PER_SAMPLE_SIZE = 20
DETECTION_INDEX = int(WORDS_PER_SAMPLE_SIZE / 2)
LABELS_COUNT = 2
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
SAVE_SAMPLED = False

# cat europarl-v7.en.clean.txt |grep -o '[,\.!?]'|wc -l
# 5102642
# cat europarl-v7.en.clean.txt |wc -w
# 53603063
# 53603063 / 5102642 = 10.5

# How large vocab? http://conversationsdirect.com/index.php?option=com_content&view=article&id=142%3Ahow-many-words-do-you-need-to-know-to-understand-english&catid=68%3Aarticles&Itemid=149&lang=en
# 2^13 = 8192
VOCAB_SIZE = 8192


# Clean and label data

def cleanData():
    toDelete = re.compile(
        '\([^)]*\)|.*Resumption of the session.*|.*VOTE.*|^Agenda$|.*report[ ]*$|^$|^\.$|([^)]*)|[^a-z0-9A-Z\',\.?! ]')
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

    print("Sampling data...")
    SAMPLE_COUNT = 10000000
    LOG_SAMPLE_NUM_STEP = 10000
    DOT_LIKE = re.compile('.*[,;.!?]')

    def incrementSampleNum(sampleNum):
        sampleNum += 1
        if sampleNum % LOG_SAMPLE_NUM_STEP == 0:
            print('sampleNum: ' + str(sampleNum))
        return sampleNum

    def readwords(mfile):
        byte_stream = itertools.groupby(
            itertools.takewhile(lambda c: bool(c),
                                map(mfile.read,
                                    itertools.repeat(1))), str.isspace)

        return ("".join(group) for pred, group in byte_stream if not pred)


    with open(BASE_DIR + "/europarl-v7/europarl-v7.en.samples.txt", 'w', encoding="utf8") as output:
        with open(BASE_DIR + "/europarl-v7/europarl-v7.en.clean.txt", 'r', encoding="utf8") as input:
            window = []
            sampleNum = 0
            for word in readwords(input):
                if len(window) < WORDS_PER_SAMPLE_SIZE:
                    window.append(word)
                    continue
                window.append(word)
                window.pop(0)
                middle = window[-DETECTION_INDEX]
                if DOT_LIKE.match(middle) is not None:
                    label = True
                else:
                    label = False
                output.write(' '.join(window))
                output.write(' ' + str(label))
                output.write('\n')
                sampleNum = incrementSampleNum(sampleNum)
                if sampleNum > SAMPLE_COUNT:
                    break
                if label:
                    window = window[-DETECTION_INDEX + 1:]




def loadSamples():
    SAMPLE_COUNT = 100000000000
    print('Loading maximum ' + str(SAMPLE_COUNT) + ' samples')
    with open(BASE_DIR + "/europarl-v7/europarl-v7.en.samples.txt", 'r', encoding="utf8") as input:
        samples = []
        labels = []
        for fullLine in input:
            line = fullLine.rstrip()
            split = line.split(' ')
            samples.append(' '.join(split[:-1]))
            labels.append(bool(split[-1]))
            if len(samples) > SAMPLE_COUNT:
                break
        return labels, samples


def tokenize(labels, samples):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(samples)
    tokenized_samples = tokenizer.texts_to_sequences(samples)
    padded_samples = pad_sequences(tokenized_samples, maxlen=WORDS_PER_SAMPLE_SIZE)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    tokenized_labels = to_categorical(np.asarray(labels))

    print('Shape of padded_samples tensor:', padded_samples.shape)
    print('Shape of tokenized_labels tensor:', tokenized_labels.shape)

    return tokenized_labels, padded_samples, tokenizer


# split the data into a training set and a validation set
def splitTrainingAndValidation(labels, samples):
    indices = np.arange(samples.shape[0])
    np.random.shuffle(indices)
    samples = samples[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * samples.shape[0])

    x_train = samples[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = samples[-nb_validation_samples:]
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
    found = 0
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            found += 1
    print("Found " + str(found) + " words in embeddings.")
    return embedding_matrix


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
def createEmbeddingLayer(tokenizer):
    embeddings_index = indexEmbeddingWordVectors()
    nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index))
    embedding_matrix = prepareEmbeddingMatrix(tokenizer.word_index, embeddings_index, nb_words)
    return Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=WORDS_PER_SAMPLE_SIZE,
                                trainable=False, input_shape=(WORDS_PER_SAMPLE_SIZE,))


def createModel(tokenizer):
    print('Creating model.')
    model = Sequential()
    model.add(createEmbeddingLayer(tokenizer))
    model.add(Conv1D(512, 4, activation='relu'))
    model.add(Dropout(0.25))
    # model.add(Conv1D(2048, 4, activation='relu', strides=2))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(LABELS_COUNT, activation='softmax'))
    # alternative optimizer: rmsprop, adam
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


def trainModel(model, x_train, y_train, x_val, y_val):
    print("Training")
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=4, batch_size=128)
    # model.save_weights(BASE_DIR + "/europarl-v7/europarl-v7.en.model")
    return model


def customTest(model, tokenizer, samples):
    printSampleEvaluation(model, tokenizer, 'Altman was named president of Y Combinator, which funded the startup he co-founded in the first batch of funded companies in 2005.')
    printSampleEvaluation(model, tokenizer, 'In a 2014 blog post, Altman stated that the total valuation of all Y Combinator companies had surpassed $65 billion, including well-known companies like Airbnb, Dropbox, Zenefits and Stripe.')
    printSampleEvaluation(model, tokenizer, 'In September 2016 Altman announced that he will be president of YC Group, which includes Y Combinator and other units.')
    for sample in samples[:100]:
        printSampleEvaluation(model, tokenizer, sample)


def printSampleEvaluation(model, tokenizer, sample):
    tokenized = pad_sequences(tokenizer.texts_to_sequences(sample), maxlen=WORDS_PER_SAMPLE_SIZE)
    preds = list(model.predict(tokenized)[0])
    index = preds.index(max(preds))
    for i, word in enumerate(sample.split(' ')):
        print(word, end=' ')
        if i == DETECTION_INDEX:
            if index == 0:
                print('*', end=' ')
            else:
                print('', end=' ')
    if index == WORDS_PER_SAMPLE_SIZE:
        print(" *None*", end=' ')
    print('')


def main():
    # cleanData()
    sampleData()
    labels, samples = loadSamples()
    tokenized_labels, tokenized_samples, tokenizer = tokenize(labels, samples)
    x_train, y_train, x_val, y_val = splitTrainingAndValidation(tokenized_labels, tokenized_samples)
    model = createModel(tokenizer)
    trainModel(model, x_train, y_train, x_val, y_val)
    customTest(model, tokenizer, samples)


main()
