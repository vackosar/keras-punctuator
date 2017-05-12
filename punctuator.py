#!/home/ubuntu/anaconda3/bin/python

'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

Europarl v7 
http://hltshare.fbk.eu/IWSLT2012/training-parallel-europarl.tgz

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

BASE_DIR = '/data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
EURO_PARL_DIR = os.path.join(BASE_DIR, 'europarl')
PUNCTUATOR_DIR = os.path.join(BASE_DIR, 'punctuator')
FREEZE_DIR = os.path.join(PUNCTUATOR_DIR, 'freezed')
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

def cleanData(inputFile=os.path.join(EURO_PARL_DIR, 'europarl-v7.en')):
    sys.stderr.write("Cleaning data " + inputFile + "\n")
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
        sampleCount=3000000,
        inputFile=os.path.join(EURO_PARL_DIR, "europarl-v7.en.clean.txt"),
        outputFile=os.path.join(EURO_PARL_DIR, "europarl-v7.en.samples.txt"),
        weighted=True,
        testPercentage=0.8):
    import itertools
    from random import randint

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


def loadSamples(samplesCount, source=os.path.join(EURO_PARL_DIR, 'europarl-v7.en.samples.txt')):
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
    np.save(os.path.join(PUNCTUATOR_DIR, name + '.npy'), obj)

def loadObject(name):
    """ :rtype: dict """
    return np.load(os.path.join(PUNCTUATOR_DIR, name + '.npy')).item()


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
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


def trainModel(model, xTrain, yTrain, xVal, yVal):
    sys.stderr.write("Training" + "\n")
    EPOCHS = 1
    for i in range(0, EPOCHS):
        model.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=1, batch_size=128)
        model.save_weights(os.path.join(PUNCTUATOR_DIR, "model"))
        test()
    return model


def test(file=os.path.join(EURO_PARL_DIR, 'europarl-v7.en.samples.txt.test')):
    labels, samples = loadSamples(100000, file)
    wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(os.path.join(PUNCTUATOR_DIR, "model"))
    tokenizedLabels, tokenizedSamples = tokenize(labels, samples, wordIndex)
    sys.stderr.write("Was: ['loss', 'acc']: [0.23739479177507825, 0.9305806942025947]" + "\n")
    metrics_values = model.evaluate(tokenizedSamples, tokenizedLabels, 128)
    sys.stderr.write(str(model.metrics_names) + ': ' + str(metrics_values) + "\n")
    punctuate(samples[:500], wordIndex, model, file)

def punctuateFile(file):
    cleanFile = cleanData(file)
    sampledFile = cleanFile + ".sampled"
    labels, samples = sampleData(10000000, cleanFile, sampledFile, False, 1)
    wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(os.path.join(PUNCTUATOR_DIR, "model"))
    punctuatedFile = file + '.punctuated.txt'
    punctuate(samples, wordIndex, model, punctuatedFile)


def punctuate(samples, wordIndex, model, punctuatedFilePrefix):
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
    punctuatedFile = punctuatedFilePrefix + '.punct.txt'
    with open(punctuatedFile, 'w', encoding="utf8") as output:
        for sample in samples:
            sequences = texts_to_sequences(wordIndex, [sample], MAX_NB_WORDS)
            tokenized = pad_sequences(sequences, maxlen=WORDS_PER_SAMPLE_SIZE)
            preds = list(model.predict(tokenized)[0])
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

    processed = punctuatedFilePrefix + '.p.txt'
    postProcess(punctuatedFile, processed)
    with open(processed, encoding="utf8") as input:
        for fullLine in input:
            print(fullLine)


def saveWithSavedModel():
    # K.set_learning_phase(0)  # all new operations will be in test mode from now on

    # wordIndex = loadWordIndex()
    model = createModel()
    model.load_weights(os.path.join(PUNCTUATOR_DIR, "model"))


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
    checkpoint_prefix = os.path.join(FREEZE_DIR, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"
    saver_write_version = 1

    # We'll create an input graph that has a single variable containing 1.0,
    # and that then multiplies it by 2.
    from tensorflow.python.framework import ops
    with ops.Graph().as_default():
        from tensorflow.python.ops import variables
        model = createModel()
        model.load_weights(os.path.join(PUNCTUATOR_DIR, "model"))
        from tensorflow.python.client import session
        sess = session.Session()
        init = variables.global_variables_initializer()
        sess.run(init)
        # output = sess.run(output_node)
        # self.assertNear(2.0, output, 0.00001)
        from tensorflow.python.training import saver as saver_lib
        saver = saver_lib.Saver(write_version=saver_write_version)
        checkpoint_path = saver.save(
            sess,
            checkpoint_prefix,
            global_step=0,
            latest_filename=checkpoint_state_name)
        from tensorflow.python.framework import graph_io
        graph_io.write_graph(sess.graph, FREEZE_DIR, input_graph_name)
        sess.close()

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(FREEZE_DIR, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    output_node_names = model.output.name.split(':')[0]
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(FREEZE_DIR, output_graph_name)
    clear_devices = False

    from tensorflow.python.tools import freeze_graph
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")

    exportWordIndex(loadWordIndex())

def testFreezed():
    import tensorflow as tf
    from tensorflow import import_graph_def
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        with gfile.FastGFile(os.path.join(FREEZE_DIR, "output_graph.pb"),'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            x = tf.placeholder(tf.int32, shape=[1, 30], name="input")
            import_graph_def(graph_def, input_map={"embedding_1_input": x})
        # input_x = sess.graph.get_tensor_by_name("import/embedding_1_input:0")
        # print(input_x)
        # output = sess.graph.get_tensor_by_name("import/dense_1/Softmax:0")
        # print(output)
        feed = np.array([1981, 12531, 12, 209, 42, 360, 7212, 96, 19999, 796, 3, 10, 8841, 7481, 7228, 464, 42, 177, 19999, 362, 425, 3, 2191, 206, 3, 19, 42, 132, 17094, 60], ndmin=2)
        print(sess.run("import/dense_1/Softmax:0", {x: feed}))


def exportWordIndex(wordIndex):
    '''
    :param wordIndex: dict 
    :return: None
    '''

    outputFile = os.path.join(PUNCTUATOR_DIR, 'freezed', 'output_word_index')
    with open(outputFile, 'w', encoding="utf8") as output:
        for item in wordIndex.items():
            output.write(item[0] + " " + str(item[1]) + '\n')


def main():
    # cleanData()
    # labels, samples = sampleData(5000000, weighted=False)
    # labels, samples = loadSamples(5000000)
    # wordIndex = saveWordIndex(samples)
    # wordIndex = loadWordIndex()
    # tokenizedLabels, tokenizedSamples = tokenize(labels, samples, wordIndex)
    # xTrain, yTrain, xVal, yVal = splitTrainingAndValidation(tokenizedLabels, tokenizedSamples)
    # model = createModel(wordIndex)
    # trainModel(model, xTrain, yTrain, xVal, yVal)
    # test()
    # punctuateFile(os.path.join(EURO_PARL_DIR, 'advice.txt'))
    # punctuateFile(os.path.join(EURO_PARL_DIR, 'musk.txt'))
    # saveWithSavedModel()
    # freeze()
    testFreezed()
    sys.stderr.write("Done")

if len(sys.argv) == 2:
    file = sys.argv[1]
    punctuateFile(file)
else:
    main()
