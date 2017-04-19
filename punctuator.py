'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
'''

from __future__ import print_function

import os
import re
from collections import OrderedDict

import numpy as np

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, Embedding
from keras.models import Sequential

BASE_DIR = 'D:\\IdeaProjects\\data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
DOT_LIKE = ',;.!?'
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

def cleanData(inputFile='europarl-v7.en'):
    print("Cleaning data " + inputFile)
    mappings = OrderedDict([
        (re.compile("['’]"), "'"),
        (re.compile("' s([" + DOT_LIKE + " ])"), "'s\g<1>"),
        (re.compile(" '([^" + DOT_LIKE + "']*)'"), ' \g<1>'),
        (re.compile("'"), " '"),
        (re.compile('\([^)]*\)'), ''),
        (re.compile('[-—]'), ' '),
        (re.compile('[^a-z0-9A-Z\',\.?! ]'), ''),
        (re.compile('^$|^\.$'), ''),
        (re.compile('.*Resumption of the session.*|.*VOTE.*|^Agenda$.*report[ ]*$'), ''),
    ])
    with open(BASE_DIR + "/europarl-v7/" + inputFile + '.clean.txt', 'w', encoding="utf8") as output:
        with open(BASE_DIR + "/europarl-v7/" + inputFile, encoding="utf8") as input:
            for fullLine in input:
                line = fullLine.rstrip()
                for pattern, replacement in mappings.items():
                    line = pattern.sub(replacement, line)
                if len(line) == 0:
                    continue
                output.write(line + " ")


def sampleData(sampleCount=3000000, inputFile="europarl-v7.en.clean.txt", outputFile="europarl-v7.en.samples.txt", weighted=True):
    import itertools
    from random import randint

    print("Sampling data " + inputFile + ' into ' + outputFile)
    LOG_SAMPLE_NUM_STEP = 10000
    DOT_LIKE_REGEX = re.compile('.*[' + DOT_LIKE + ']')

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

    def samplingTestValues(sampleNum, sampleCount):
        return int(sampleCount * 0.8) < sampleNum

    def write(output, window, label):
        output.write(' '.join(window))
        output.write(' ' + str(label))
        output.write('\n')

    with open(BASE_DIR + "/europarl-v7/" + outputFile, 'w', encoding="utf8") as output:
        with open(BASE_DIR + "/europarl-v7/" + outputFile + ".test", 'w', encoding="utf8") as testOutput:
            with open(BASE_DIR + "/europarl-v7/" + inputFile, 'r', encoding="utf8") as input:
                window = []
                sampleNum = 0
                for word in readwords(input):
                    if len(window) < WORDS_PER_SAMPLE_SIZE:
                        window.append(word)
                        continue
                    window.append(word)
                    window.pop(0)
                    middle = window[-DETECTION_INDEX]
                    if DOT_LIKE_REGEX.match(middle) is not None:
                        label = True
                    else:
                        label = False
                        if not samplingTestValues(sampleNum, sampleCount) and randint(0, 9) < 9:
                            continue
                    if samplingTestValues(sampleNum, sampleCount):
                        write(testOutput, window, label)
                    else:
                        write(output, window, label)
                    sampleNum = incrementSampleNum(sampleNum)
                    if sampleNum > sampleCount:
                        break


def loadSamples(samplesCount, source='europarl-v7.en.samples.txt'):
    print('Loading maximum ' + str(samplesCount) + ' samples from ' + source)
    with open(BASE_DIR + "/europarl-v7/" + source, 'r', encoding="utf8") as input:
        samples = []
        labels = []
        for fullLine in input:
            line = fullLine.rstrip()
            split = line.split(' ')
            samples.append(' '.join(split[:-1]))
            if split[-1] == "True":
                labels.append(True)
            else:
                labels.append(False)
            if len(samples) > samplesCount:
                break
        return labels, samples


def texts_to_sequences(word_index, texts, num_words):
    lastWord = num_words
    sequences = []
    for text in texts:
        seq = text_to_word_sequence(text)
        vect = []
        for w in seq:
            i = word_index.get(w)
            if i is not None:
                if num_words and i >= num_words:
                    vect.append(lastWord)
                else:
                    vect.append(i)
            else:
                vect.append(lastWord)
        sequences.append(vect)
    return sequences

def loadWordIndex():
    return loadObject('word_index')

def saveWordIndex(samples):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(samples)
    word_index = tokenizer.word_index
    saveObject(word_index, 'word_index')
    print('Found %s unique tokens.' % len(word_index))
    return word_index


def tokenize(labels, samples, word_index):

    tokenizedSamples = texts_to_sequences(word_index, samples, MAX_NB_WORDS)
    padded_samples = pad_sequences(tokenizedSamples, maxlen=WORDS_PER_SAMPLE_SIZE)

    tokenized_labels = to_categorical(np.asarray(labels))

    print('Shape of padded_samples tensor:', padded_samples.shape)
    print('Shape of tokenized_labels tensor:', tokenized_labels.shape)

    return tokenized_labels, padded_samples

def saveObject(obj, name):
    np.save(BASE_DIR + '/europarl-v7/'+ name + '.npy', obj)

def loadObject(name):
    """

    :rtype: dict
    """
    return np.load(BASE_DIR + '/europarl-v7/'+ name + '.npy').item()


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
def createEmbeddingLayer(word_index=None):
    if word_index is None:
        return Embedding(MAX_NB_WORDS,
                              EMBEDDING_DIM,
                              input_length=WORDS_PER_SAMPLE_SIZE,
                              trainable=False, input_shape=(WORDS_PER_SAMPLE_SIZE,))
    else:
        embeddings_index = indexEmbeddingWordVectors()
        embedding_matrix = prepareEmbeddingMatrix(word_index, embeddings_index, MAX_NB_WORDS)
        return Embedding(MAX_NB_WORDS,
                         EMBEDDING_DIM,
                         input_length=WORDS_PER_SAMPLE_SIZE,
                         weights=[embedding_matrix],
                         trainable=False, input_shape=(WORDS_PER_SAMPLE_SIZE,))


def createModel(word_index=None):
    print('Creating model.')
    model = Sequential()
    model.add(createEmbeddingLayer(word_index))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(LABELS_COUNT, activation='softmax'))
    # alternative optimizer: rmsprop, adam
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


def trainModel(model, x_train, y_train, x_val, y_val):
    print("Training")
    EPOCHS = 2
    for i in range(0, EPOCHS):
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=128)
        model.save_weights(BASE_DIR + "/europarl-v7/europarl-v7.en.model")
        test()
    return model


def test():
    labels, samples = loadSamples(15000, 'europarl-v7.en.samples.txt.test')
    word_index = loadWordIndex()
    model = createModel()
    model.load_weights(BASE_DIR + "/europarl-v7/europarl-v7.en.model")
    tokenized_labels, tokenized_samples = tokenize(labels, samples, word_index)
    print("Was: ['loss', 'acc']: [0.24365297650648796, 0.90473968397735816]")
    metrics_values = model.evaluate(tokenized_samples, tokenized_labels, 128)
    print(str(model.metrics_names) + ': ' + str(metrics_values))
    for sample in samples[:10]:
        printSampleEvaluation(model, word_index, sample)


def printSampleEvaluation(model, word_index, sample):
    sequences = texts_to_sequences(word_index, [sample], MAX_NB_WORDS)
    tokenized = pad_sequences(sequences, maxlen=WORDS_PER_SAMPLE_SIZE)
    preds = list(model.predict(tokenized)[0])
    index = preds.index(max(preds))
    for i, word in enumerate(sample.split(' ')):
        print(word, end=' ')
        if i == DETECTION_INDEX:
            if index == 0:
                print('*', end=' ')
            else:
                print('|', end=' ')
    if index == WORDS_PER_SAMPLE_SIZE:
        print(" *None*", end=' ')
    print('')


def main():
    # cleanData()
    # sampleData(1500000)
    labels, samples = loadSamples(1500000)
    # saveWordIndex(samples)
    word_index = loadWordIndex()
    tokenized_labels, tokenized_samples = tokenize(labels, samples, word_index)
    x_train, y_train, x_val, y_val = splitTrainingAndValidation(tokenized_labels, tokenized_samples)
    model = createModel(word_index)
    trainModel(model, x_train, y_train, x_val, y_val)
    test()

main()
