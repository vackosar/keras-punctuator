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
import sys

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, Embedding
from keras.models import Sequential

BASE_DIR = 'D:\\IdeaProjects\\data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
DOT_LIKE = ',;.!?'
DOT_LIKE_AND_SPACE = ',;.!? '
WORDS_PER_SAMPLE_SIZE = 30
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

def cleanData(inputFile=os.path.join(BASE_DIR, 'europarl-v7.en')):
    print("Cleaning data " + inputFile)
    mappings = OrderedDict([
        (re.compile("['’]"), "'"),
        (re.compile("' s([" + DOT_LIKE_AND_SPACE + "])"), "'s\g<1>"),
        (re.compile("n't"), " n't"),
        (re.compile(" '([^" + DOT_LIKE + "']*)'"), ' \g<1>'),
        (re.compile("'([^t])"), " '\g<1>"),
        (re.compile('\([^)]*\)'), ''),
        (re.compile('[-—]'), ' '),
        (re.compile('[^a-z0-9A-Z\',\.?! ]'), ''),
        (re.compile('^$|^\.$'), ''),
        (re.compile('.*Resumption of the session.*|.*VOTE.*|^Agenda$.*report[ ]*$'), ''),
    ])
    cleanFile = inputFile + '.clean.txt'
    with open(cleanFile, 'w', encoding="utf8") as output:
        with open(inputFile, encoding="utf8") as input:
            for fullLine in input:
                line = fullLine.rstrip()
                for pattern, replacement in mappings.items():
                    line = pattern.sub(replacement, line)
                if len(line) == 0:
                    continue
                output.write(line + " ")
    return cleanFile


def sampleData(
        sampleCount=3000000,
        inputFile=os.path.join(BASE_DIR, "europarl-v7", "europarl-v7.en.clean.txt"),
        outputFile=os.path.join(BASE_DIR, "europarl-v7", "europarl-v7.en.samples.txt"),
        weighted=True,
        testPercentage=0.8):
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

    def samplingTestValues(sampleNum, sampleCount, testPercentage=0.8):
        return int(sampleCount * testPercentage) < sampleNum

    def write(output, window, label):
        output.write(' '.join(window))
        output.write(' ' + str(label))
        output.write('\n')

    def skipNonDotSample(weighted):
        DOT_WEIGHT = 1
        """ Skip non dot samples to prevent local minima of no dots. """
        return weighted and randint(0, 9) < DOT_WEIGHT

    def skip(weighted):
        """ Skips for more diverse input. """
        return weighted and randint(0, 9) < 3

    samples = []
    labels = []
    samplingTestValues = False
    with open(outputFile, 'w', encoding="utf8") as output:
        with open(outputFile + ".test", 'w', encoding="utf8") as testOutput:
            with open(inputFile, 'r', encoding="utf8") as input:
                window = []
                sampleNum = 0
                for word in readwords(input):
                    if len(window) < WORDS_PER_SAMPLE_SIZE:
                        window.append(word)
                        continue
                    if sampleNum != 0:
                        window.append(word)
                        window.pop(0)
                    middle = window[-DETECTION_INDEX]
                    if skip(weighted):
                        continue
                    if DOT_LIKE_REGEX.match(middle) is not None:
                        label = True
                    else:
                        label = False
                    if samplingTestValues:
                        write(testOutput, window, label)
                    else:
                        samples.append(' '.join(window))
                        labels.append(label)
                        write(output, window, label)
                    sampleNum = incrementSampleNum(sampleNum)
                    if int(sampleCount * testPercentage) < sampleNum + 1:
                        samplingTestValues = True
                        weighted = False
                    if 1 + sampleNum > sampleCount:
                        break
    return labels, samples


def loadSamples(samplesCount, source=os.path.join(BASE_DIR, "europarl-v7", 'europarl-v7.en.samples.txt')):
    print('Loading maximum ' + str(samplesCount) + ' samples from ' + source)
    with open(source, 'r', encoding="utf8") as input:
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


def texts_to_sequences(wordIndex, texts, num_words):
    lastWord = num_words - 1
    sequences = []
    for text in texts:
        seq = text_to_word_sequence(text)
        vect = []
        for w in seq:
            i = wordIndex.get(w)
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
    return loadObject('wordIndex')

def saveWordIndex(samples):
    print('Building word index.')
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(samples)
    wordIndex = tokenizer.word_index
    saveObject(wordIndex, 'wordIndex')
    print('Found %s unique tokens.' % len(wordIndex))
    return wordIndex


def tokenize(labels, samples, wordIndex):
    print('Tokenizing samples.')

    tokenizedSamples = texts_to_sequences(wordIndex, samples, MAX_NB_WORDS)
    paddedSamples = pad_sequences(tokenizedSamples, maxlen=WORDS_PER_SAMPLE_SIZE)

    tokenizedLabels = to_categorical(np.asarray(labels))

    print('Shape of paddedSamples tensor:', paddedSamples.shape)
    print('Shape of tokenizedLabels tensor:', tokenizedLabels.shape)

    return tokenizedLabels, paddedSamples

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

    xTrain = samples[:-nb_validation_samples]
    yTrain = labels[:-nb_validation_samples]
    xVal = samples[-nb_validation_samples:]
    yVal = labels[-nb_validation_samples:]
    return xTrain, yTrain, xVal, yVal


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


def prepareEmbeddingMatrix(wordIndex, embeddings_index, nb_words):
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    found = 0
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in wordIndex.items():
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
def createEmbeddingLayer(wordIndex=None):
    if wordIndex is None:
        return Embedding(MAX_NB_WORDS,
                              EMBEDDING_DIM,
                              input_length=WORDS_PER_SAMPLE_SIZE,
                              trainable=False, input_shape=(WORDS_PER_SAMPLE_SIZE,))
    else:
        embeddings_index = indexEmbeddingWordVectors()
        embedding_matrix = prepareEmbeddingMatrix(wordIndex, embeddings_index, MAX_NB_WORDS)
        return Embedding(MAX_NB_WORDS,
                         EMBEDDING_DIM,
                         input_length=WORDS_PER_SAMPLE_SIZE,
                         weights=[embedding_matrix],
                         trainable=False, input_shape=(WORDS_PER_SAMPLE_SIZE,))


def createModel(wordIndex=None):
    print('Creating model.')
    model = Sequential()
    model.add(createEmbeddingLayer(wordIndex))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(LABELS_COUNT, activation='softmax'))
    # alternative optimizer: rmsprop, adam
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


def trainModel(model, xTrain, yTrain, xVal, yVal):
    print("Training")
    EPOCHS = 1
    for i in range(0, EPOCHS):
        model.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=1, batch_size=128)
        model.save_weights(os.path.join(BASE_DIR, "europarl-v7", "europarl-v7.en.model"))
        test()
    return model


def test(file=os.path.join(BASE_DIR, "europarl-v7", 'europarl-v7.en.samples.txt.test')):
    labels, samples = loadSamples(100000, file)
    wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(os.path.join(BASE_DIR, "europarl-v7/europarl-v7.en.model"))
    tokenizedLabels, tokenizedSamples = tokenize(labels, samples, wordIndex)
    print("Was: ['loss', 'acc']: [0.26865265584550135, 0.92189078111007028]")
    metrics_values = model.evaluate(tokenizedSamples, tokenizedLabels, 128)
    print(str(model.metrics_names) + ': ' + str(metrics_values))
    print(punctuate(samples[:500], wordIndex, model))


def punctuateFile(file):
    cleanFile = cleanData(file)
    sampledFile = cleanFile + ".sampled"
    labels, samples = sampleData(10000000, cleanFile, sampledFile, False, 1)
    wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(os.path.join(BASE_DIR, "europarl-v7", "europarl-v7.en.model"))
    text = punctuate(samples, wordIndex, model)
    print(text)
    with open(file + '.punctuated.txt', 'w', encoding="utf8") as output:
        output.write(text)


def punctuate(samples, wordIndex, model):
    firstSample = samples[0].split(' ')
    lastSample = samples[len(samples) - 1].split(' ')

    for i in range(1, DETECTION_INDEX + 1):
        preSample = lastSample[WORDS_PER_SAMPLE_SIZE - i:] + firstSample[:WORDS_PER_SAMPLE_SIZE - i]
        samples.insert(0, ' '.join(preSample))
        if i != DETECTION_INDEX:
            postSample = lastSample[i:] + firstSample[:i]
            samples.append(' '.join(postSample))

    DOT_LIKE_REGEX = re.compile('[' + DOT_LIKE + ']')
    capitalize = True
    text = ''
    for sample in samples:
        sequences = texts_to_sequences(wordIndex, [sample], MAX_NB_WORDS)
        tokenized = pad_sequences(sequences, maxlen=WORDS_PER_SAMPLE_SIZE)
        preds = list(model.predict(tokenized)[0])
        index = preds.index(max(preds))
        punctuatedWord = sample.split(' ')[DETECTION_INDEX]
        word = DOT_LIKE_REGEX.sub('', punctuatedWord).lower()
        if capitalize:
            text += word.capitalize()
        else:
            text += word
        if index == 1:
            text += '. '
            capitalize = True
        else:
            text += ' '
            capitalize = False
    return text


def main():
    # cleanData()
    # labels, samples = sampleData(20000000, weighted=False)
    labels, samples = loadSamples(10000000)
    wordIndex = saveWordIndex(samples)
    # wordIndex = loadWordIndex()
    tokenizedLabels, tokenizedSamples = tokenize(labels, samples, wordIndex)
    xTrain, yTrain, xVal, yVal = splitTrainingAndValidation(tokenizedLabels, tokenizedSamples)
    model = createModel(wordIndex)
    trainModel(model, xTrain, yTrain, xVal, yVal)
    # test()
    punctuateFile(os.path.join(BASE_DIR, "europarl-v7", 'ted-ai.txt'))
    punctuateFile(os.path.join(BASE_DIR, "europarl-v7", 'advice.txt'))

if len(sys.argv) == 2:
    file = sys.argv[1]
    punctuateFile(file)
else:
    main()
