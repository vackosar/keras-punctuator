#!/usr/bin/env python

'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

Europarl v7 
http://hltshare.fbk.eu/IWSLT2012/training-parallel-europarl.tgz

News Crawl from WMT 2012 (en, fr), 7GB
http://hltshare.fbk.eu/IWSLT2012/training-monolingual-newsshuffled.tgz

Additional data
http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html
'''

from __future__ import print_function

import os
import re

import shutil
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
import keras.backend as K

BASE_DIR = '/data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
NEWS_DIR = os.path.join(BASE_DIR, 'training-monolingual-newsshuffled')
PUNCTUATOR_DIR = os.path.join(BASE_DIR, 'punctuator')
MODEL_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model-data")
TMP_DIR = "tmp"
KERAS_WEIGHTS_FILE = os.path.join(MODEL_DATA_DIR, "model")
DOT_LIKE = ',;.!?'
DOT_LIKE_AND_SPACE = ',;.!? '
WORDS_PER_SAMPLE_SIZE = 30
DETECTION_INDEX = int(WORDS_PER_SAMPLE_SIZE / 2)
LABELS_COUNT = 2
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
SAVE_SAMPLED = False

# cat europarl-v7.en.clean |grep -o '[,\.!?]'|wc -l
# 5102642
# cat europarl-v7.en.clean |wc -w
# 53603063
# 53603063 / 5102642 = 10.5

# How large vocab? http://conversationsdirect.com/index.php?option=com_content&view=article&id=142%3Ahow-many-words-do-you-need-to-know-to-understand-english&catid=68%3Aarticles&Itemid=149&lang=en
# 2^13 = 8192
VOCAB_SIZE = 8192

def cleanData(inputFile):
    sys.stderr.write("Cleaning data " + inputFile + "\n")
    mappings = OrderedDict([
        (re.compile("['’]"), "'"),
        # (re.compile("' s([" + DOT_LIKE_AND_SPACE + "])"), "'s\g<1>"), # Removes strange text mistake pattern in europarl data.
        (re.compile("n't"), " n't"),
        #(re.compile(" '([^" + DOT_LIKE + "']*)'"), '. \g<1>.'), # Remove quoting apostrophes.
        (re.compile("'([^t])"), " '\g<1>"), # Separate tokens like "'s" "'ll" and so on.
        #(re.compile('\([^)]*\)'), ''), # Removes bracketed.
        (re.compile('[-—]'), ' '), # Dash to space.
        (re.compile('[^a-z0-9A-Z\',\.?! ]'), ''), # Other unknown to nothing.
        # (re.compile('^$|^\.$'), ''), # Removes empty line.
    ])
    cleanFile = inputFile + '.clean'
    regexProcess(mappings, inputFile, cleanFile)
    return cleanFile

def postProcess(inputFile, outputFile):
    sys.stderr.write("Post processing " + inputFile + "\n")
    mappings = OrderedDict([
        (re.compile(" n't"), "n't"),
        (re.compile(" '"), "'"),
    ])
    regexProcess(mappings, inputFile, outputFile)

def regexProcess(mappings, inputFile, outputFile):
    with open(outputFile, 'w', encoding="utf8") as output:
        with open(inputFile, encoding="utf8") as input:
            for fullLine in input:
                line = fullLine.rstrip()
                for pattern, replacement in mappings.items():
                    line = pattern.sub(replacement, line)
                if len(line) == 0:
                    continue
                output.write(line + " ")
    return outputFile

def sampleData(
        sampleCount,
        inputFile,
        weighted=True,
        testPercentage=0.8):
    import itertools
    from random import randint

    outputFile = inputFile + ".samples"
    sys.stderr.write("Sampling data " + inputFile + ' into ' + outputFile + "\n")
    LOG_SAMPLE_NUM_STEP = 10000
    DOT_LIKE_REGEX = re.compile('.*[' + DOT_LIKE + ']')

    def incrementSampleNum(sampleNum):
        sampleNum += 1
        if sampleNum % LOG_SAMPLE_NUM_STEP == 0:
            sys.stderr.write('sampleNum: ' + str(sampleNum) + "\n")
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


def loadSamples(samplesCount, source):
    sys.stderr.write('Loading maximum ' + str(samplesCount) + ' samples from ' + source + "\n")
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
    sys.stderr.write('Building word index.' + "\n")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(samples)
    wordIndex = {}
    for num, item in enumerate(tokenizer.word_index.items()):
        if num >= MAX_NB_WORDS - 1:
            break
        else:
            wordIndex[item[0]] = item[1]
    saveObject(wordIndex, 'wordIndex')
    sys.stderr.write('Found %s unique tokens.' % len(wordIndex) + "\n")
    return wordIndex


def tokenize(labels, samples, wordIndex):
    sys.stderr.write('Tokenizing samples.' + "\n")

    tokenizedSamples = texts_to_sequences(wordIndex, samples, MAX_NB_WORDS)
    paddedSamples = pad_sequences(tokenizedSamples, maxlen=WORDS_PER_SAMPLE_SIZE)

    tokenizedLabels = to_categorical(np.asarray(labels))

    sys.stderr.write('Shape of paddedSamples tensor:' + str(paddedSamples.shape) + "\n")
    sys.stderr.write('Shape of tokenizedLabels tensor:' + str(tokenizedLabels.shape) + "\n")

    return tokenizedLabels, paddedSamples

def saveObject(obj, name):
    np.save(os.path.join(MODEL_DATA_DIR, name + '.npy'), obj)

def loadObject(name):
    """ :rtype: dict """
    return np.load(os.path.join(MODEL_DATA_DIR, name + '.npy')).item()


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
    sys.stderr.write('Indexing word vectors.' + "\n")
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    sys.stderr.write('Found %s word vectors.' % len(embeddings_index) + "\n")
    return embeddings_index


def prepareEmbeddingMatrix(wordIndex, embeddings_index, nb_words):
    sys.stderr.write('Preparing embedding matrix.' + "\n")
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
    sys.stderr.write("Found " + str(found) + " words in embeddings." + "\n")
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
    sys.stderr.write('Creating model.' + "\n")
    model = Sequential()
    model.add(createEmbeddingLayer(wordIndex))
    model.add(Conv1D(512, 3, activation='relu'))
    if wordIndex is not None:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(LABELS_COUNT, activation='softmax'))
    # alternative optimizer: rmsprop, adam
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc', precision, recall, fbeta_score])
    return model


def trainModel(model, xTrain, yTrain, xVal, yVal, filePrefix):
    sys.stderr.write("Training" + "\n")
    EPOCHS = 1
    for i in range(0, EPOCHS):
        model.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=1, batch_size=128)
        model.save_weights(KERAS_WEIGHTS_FILE)
        test(filePrefix + ".test")
    return model


def test(file):
    labels, samples = loadSamples(100000, file)
    wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(KERAS_WEIGHTS_FILE)
    tokenizedLabels, tokenizedSamples = tokenize(labels, samples, wordIndex)
    sys.stderr.write("['loss', 'acc', 'precision', 'recall', 'fbeta_score']: [0.24152110019584505, 0.92070079298134133, 0.92070079300101071, 0.92070079300101071, 0.92070073339636593]" + "\n")
    metrics_values = model.evaluate(tokenizedSamples, tokenizedLabels, 128)
    sys.stderr.write(str(model.metrics_names) + ': ' + str(metrics_values) + "\n")
    predict = lambda tokenized: model.predict(tokenized)[0]
    punctuate(samples[:500], wordIndex, predict, file)

def punctuateFile(file):
    cleanFile = cleanData(file)
    labels, samples = sampleData(10000000, cleanFile, False, 1)
    wordIndex = loadWordIndex()
    punctuatedFile = file + '.punct'
    import tensorflow as tf
    from tensorflow import import_graph_def
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        with gfile.FastGFile(os.path.join(MODEL_DATA_DIR, "freezed.pb"),'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        import_graph_def(graph_def)
        x = sess.graph.get_tensor_by_name("import/embedding_1_input:0")
        predict = lambda tokenized: sess.run("import/dense_1/Softmax:0", {x: tokenized})[0]
        punctuate(samples, wordIndex, predict, punctuatedFile)


def punctuate(samples, wordIndex, predict, punctuatedFilePrefix):
    firstSample = samples[0].split(' ')
    lastSample = samples[len(samples) - 1].split(' ')
    filler = []
    for i in range(1, WORDS_PER_SAMPLE_SIZE):
        filler.append("however")

    for i in range(1, DETECTION_INDEX + 1):
        preSample = filler[WORDS_PER_SAMPLE_SIZE - i:] + firstSample[:WORDS_PER_SAMPLE_SIZE - i + 1]
        samples.insert(0, ' '.join(preSample))
        if i != DETECTION_INDEX:
            postSample = lastSample[i:] + filler[:i]
            samples.append(' '.join(postSample))

    DOT_LIKE_REGEX = re.compile('[' + DOT_LIKE + ']')
    capitalize = True
    punctuatedFile = punctuatedFilePrefix + '.punct'
    with open(punctuatedFile, 'w', encoding="utf8") as output:
        for sample in samples:
            sequences = texts_to_sequences(wordIndex, [sample], MAX_NB_WORDS)
            tokenized = pad_sequences(sequences, maxlen=WORDS_PER_SAMPLE_SIZE)
            preds = list(predict(tokenized))
            index = preds.index(max(preds))
            punctuatedWord = sample.split(' ')[DETECTION_INDEX]
            word = DOT_LIKE_REGEX.sub('', punctuatedWord).lower()
            if capitalize:
                output.write(word.capitalize())
            else:
                output.write(word)
            if index == 1:
                output.write('. ')
                capitalize = True
            else:
                output.write(' ')
                capitalize = False

    processed = punctuatedFile + '.proc.txt'
    postProcess(punctuatedFile, processed)
    # with open(processed, encoding="utf8") as input:
    #     for fullLine in input:
    #         print(fullLine)


def saveWithSavedModel():
    # K.set_learning_phase(0)  # all new operations will be in test mode from now on

    # wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(KERAS_WEIGHTS_FILE)


    export_path = os.path.join(PUNCTUATOR_DIR, 'graph') # where to save the exported graph

    shutil.rmtree(export_path, True)
    export_version = 1 # version number (integer)

    import tensorflow as tf
    sess = tf.Session()

    saver = tf.train.Saver(sharded=True)
    from tensorflow.contrib.session_bundle import exporter
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(input_tensor=model.input,scores_tensor=model.output)
    # model_exporter.init(sess.graph.as_graph_def(),default_graph_signature=signature)
    tf.initialize_all_variables().run(session=sess)
    # model_exporter.export(export_path, tf.constant(export_version), sess)
    from tensorflow.python.saved_model import builder as saved_model_builder
    builder = saved_model_builder.SavedModelBuilder(export_path)
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
    signature_def = predict_signature_def(
        {signature_constants.PREDICT_INPUTS: model.input},
        {signature_constants.PREDICT_OUTPUTS: model.output})
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_def
        },
        legacy_init_op=legacy_init_op)
    builder.save()


def freeze():
    checkpoint_prefix = os.path.join(TMP_DIR, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "freezed.pb"
    saver_write_version = 1

    # We'll create an input graph that has a single variable containing 1.0,
    # and that then multiplies it by 2.
    from tensorflow.python.framework import ops
    with ops.Graph().as_default():
        from keras import backend as K
        K.set_learning_phase(0)
        model = createModel()
        model.load_weights(KERAS_WEIGHTS_FILE)

        sess = K.get_session()
        from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants
        # convert_variables_to_constants(sess, sess.graph.as_graph_def(), [model.output.name.split(':')[0]])
        testGraph(sess, '')

        from tensorflow.python.training import saver as saver_lib
        saver = saver_lib.Saver(write_version=saver_write_version)
        checkpoint_path = saver.save(
            sess,
            checkpoint_prefix,
            global_step=0,
            latest_filename=checkpoint_state_name)
        from tensorflow.python.framework import graph_io
        graph_io.write_graph(sess.graph, TMP_DIR, input_graph_name)
        sess.close()


    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(TMP_DIR, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    output_node_names = model.output.name.split(':')[0]
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(MODEL_DATA_DIR, output_graph_name)
    clear_devices = False

    from tensorflow.python.tools import freeze_graph
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")

    exportWordIndex(loadWordIndex())

def getExpectedValues(feed):
    model = createModel()
    model.load_weights(KERAS_WEIGHTS_FILE)
    return model.predict(feed)

def testFreezed():
    import tensorflow as tf
    from tensorflow import import_graph_def
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        with gfile.FastGFile(os.path.join(MODEL_DATA_DIR, "freezed.pb"),'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        import_graph_def(graph_def)
        testGraph(sess, 'import/')


def testGraph(sess, prefix):
    x = sess.graph.get_tensor_by_name(prefix + "embedding_1_input:0")
    feed = np.array([1981, 12531, 12, 209, 42, 360, 7212, 96, 19999, 796, 3, 10, 8841, 7481, 7228, 464, 42, 177, 19999, 362, 425, 3, 2191, 206, 3, 19, 42, 132, 17094, 60], ndmin=2)
    actual = sess.run(prefix + "dense_1/Softmax:0", {x: feed})
    expected = getExpectedValues(feed)
    print("Expected: " + str(expected) + " Actual:" + str(actual))
    if expected[0][0] == actual[0][0] and expected[0][1] == actual[0][1]:
        print("OK")
    else:
        print("FAIL")

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def exportWordIndex(wordIndex):
    '''
    :param wordIndex: dict 
    :return: None
    '''

    outputFile = os.path.join(PUNCTUATOR_DIR, 'freezed', 'output_word_index')
    with open(outputFile, 'w', encoding="utf8") as output:
        for item in wordIndex.items():
            output.write(item[0] + " " + str(item[1]) + '\n')


def writeTensorflowDashboardLog():
    from tensorflow.python.framework import ops
    with ops.Graph().as_default():
        from keras import backend as K
        K.set_learning_phase(0)
        model = createModel()
        model.load_weights(KERAS_WEIGHTS_FILE)
        sess = K.get_session()

        testGraph(sess, '')
        import tensorflow as tf
        writer = tf.summary.FileWriter("tmp", graph=sess.graph)
        writer.flush()
        writer.close()

def halucinate():
    wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(KERAS_WEIGHTS_FILE)
    bestPred = 0
    bestSample = None
    for i in range(0, 1000000):
        sample = np.random.randint(0, 20000, (1,30))
        preds = list(model.predict(sample)[0])
        if preds[1] > bestPred:
            bestSample = sample
            bestPred = preds[1]
    print(bestPred)
    for token in bestSample[0]:
        for key, value in wordIndex.items():
            if value == token:
                print(key, end=" ")
    # Last result: 0.905726 dominance four adversity routine hold replies unsurprisingly belly imagined affirmed twice vaccine blatantly burgers rest competitions errands intimacy doing rams destruction reconciliation cia 90 groceries wasting license versus wk4 laugh Done




def main():
    dataFile = os.path.join(NEWS_DIR, 'news.2011.en.shuffled')
    # cleanData(dataFile)
    # labels, samples = sampleData(5000000, dataFile + ".clean", weighted=False)
    # labels, samples = loadSamples(5000000, dataFile + ".clean.samples")
    # wordIndex = saveWordIndex(samples)
    # wordIndex = loadWordIndex()
    # tokenizedLabels, tokenizedSamples = tokenize(labels, samples, wordIndex)
    # xTrain, yTrain, xVal, yVal = splitTrainingAndValidation(tokenizedLabels, tokenizedSamples)
    # model = createModel(wordIndex)
    # trainModel(model, xTrain, yTrain, xVal, yVal, dataFile + ".clean.samples")
    # test(dataFile + ".clean.samples.test")
    # punctuateFile(os.path.join(NEWS_DIR, 'advice.txt'))
    # punctuateFile(os.path.join(NEWS_DIR, 'musk.txt'))
    # saveWithSavedModel()
    # freeze()
    # testFreezed()
    # writeTensorflowDashboardLog()
    # halucinate()
    sys.stderr.write("Done")

if len(sys.argv) == 2:
    file = sys.argv[1]
    punctuateFile(file)
else:
    main()
