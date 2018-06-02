import numpy as np
import tensorflow as tf


def get_generator(input_file, fn=lambda line: line):
    def g():
        with open(input_file, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                yield fn(line)
    return g


def build_vocab(input_file):
    vacab = {
        "<pad>":0,
        "<oov>":1,
    }
    vacabback = {
        0:"<pad>",
        1:"<oov>",
    }
    char_set = set()
    for line in get_generator(input_file)():
        [[char_set.add(char) for char in word] for word in line.split()]

    index = 2
    for char in char_set:
        vacab[char] = index
        vacabback[index] = char
        index+=1
    return vacab, vacabback


def char2id(vocab, char):
    if char in vocab:
        return vocab[char]
    else:
        return vocab["<oov>"]


def id2char(vocabback, id):
    if id in vocabback:
        return vocabback
    else:
        return "<error>"


def get_trainning_input(params):
    def fn(line):
        words = line.split()
        chars, tags = [], []
        for word in words:
            if len(word)==1:
                chars.append(char2id(params.vocab, word))
                tags.append(params.tag["S"])
            elif len(word)==2:
                chars.append(char2id(params.vocab, word[0]))
                chars.append(char2id(params.vocab, word[1]))
                tags.append(params.tag["B"])
                tags.append(params.tag["E"])
            else:
                for i in range(len(word)):
                    chars.append(char2id(params.vocab, word[i]))
                    if i==0:
                        tags.append(params.tag["B"])
                    elif i==len(word)-1:
                        tags.append(params.tag["E"])
                    else:
                        tags.append(params.tag["M"])
        return chars, tags

    dataset = tf.data.Dataset.from_generator(get_generator(params.input, fn), (tf.int32, tf.int32),
                                             ((None,), (None,)))
    dataset = dataset.shuffle(params.buffer_size)
    dataset = dataset.repeat()

    # Append <pad> symbol
    dataset = dataset.map(
        lambda chars, tags:(
            tf.concat([[tf.constant(params.vocab["<pad>"])]*params.window_size,
                       chars, [tf.constant(params.vocab["<pad>"])]*params.window_size], axis=0),
            tf.concat([[tf.constant(params.vocab["<pad>"])]*params.window_size,
                       tags, [tf.constant(params.vocab["<pad>"])]*params.window_size], axis=0)
        ),
        num_parallel_calls=params.num_threads
    )

    # Convert to dictionary
    dataset = dataset.map(
        lambda chars, tags: {
            "chars": tf.to_int32(chars),
            "tags": tf.to_int32(tags),
            "start": tf.to_int32(tf.constant(params.window_size)),
            "end": tf.to_int32(tf.shape(chars)[0]-params.window_size)
        },
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(params.batch_size, padded_shapes={"chars": (None,), "tags": (None,),
                                                                     "start": (), "end": ()})

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


def get_evaluate_input(params):
    features = None
    return features


def get_inference_input(params):
    features = None
    return features


if __name__ == "__main__":
    pku_train = "/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/training/pku_training.utf8"
    pku_test = "/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/testing/pku_test.utf8"
    pku_gold = "/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/gold/pku_test_gold.utf8"
    params = tf.contrib.training.HParams(
        input = pku_train,
        tag = {"B":0, "M":1, "E":2, "S":3},
        vocab = None,
    )
    if params.vocab is None:
        params.vocab, params.vocabback = build_vocab(params.input)

    features = get_trainning_input(params)
    # print(features)
    with tf.Session() as sess:
        while True:
            # print([[id2char(char) for char in line] for line in sess.run(features)])
            print(sess.run(features))