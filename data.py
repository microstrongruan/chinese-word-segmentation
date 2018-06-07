import tensorflow as tf


def get_generator(input_file, fn=lambda line: line):
    def g():
        with open(input_file, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                yield fn(line)
    return g


def build_vocab_trans(input_file, tag):
    vacab = {
        "<pad>":0,
        "<oov>":1,
    }
    vacabback = {
        0:"<pad>",
        1:"<oov>",
    }
    char_set = set()

    tag_types = len(tag)
    transition=[[0 for j in range(tag_types)] for i in range(tag_types)]

    def fn(line):
        words = line.split()
        chars, tags = [], []
        for word in words:
            if len(word)==1:
                chars.append(word)
                tags.append(tag["S"])
            elif len(word)==2:
                chars.append(word[0])
                chars.append(word[1])
                tags.append(tag["B"])
                tags.append(tag["E"])
            else:
                for i in range(len(word)):
                    chars.append(word[i])
                    if i==0:
                        tags.append(tag["B"])
                    elif i==len(word)-1:
                        tags.append(tag["E"])
                    else:
                        tags.append(tag["M"])
        for i in range(len(tags)-1):
            transition[tags[i]][tags[i+1]] += 1
        return chars

    for chars in get_generator(input_file, fn)():
        for char in chars:
            char_set.add(char)

    for i in range(len(transition)):
        sum_ = sum(transition[i])
        for j in range(len(transition[i])):
            transition[i][j] = transition[i][j]/sum_

    index = 2
    for char in char_set:
        vacab[char] = index
        vacabback[index] = char
        index+=1

    return vacab, vacabback, transition


def char2id(vocab, char):
    if char in vocab:
        return int(vocab[char])
    else:
        return int(vocab["<oov>"])


def id2char(vocabback, id):
    if int(id) in vocabback:
        return vocabback[int(id)]
    elif str(id) in vocabback:
        return vocabback[str(id)]
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
        return line, chars, tags

    dataset = tf.data.Dataset.from_generator(get_generator(params.input, fn), (tf.string, tf.int32, tf.int32),
                                             ((), (None,), (None,)))
    dataset = dataset.shuffle(params.buffer_size)
    dataset = dataset.repeat()

    # Append <pad> symbol
    dataset = dataset.map(
        lambda line, chars, tags:(
            line,
            tf.concat([[tf.constant(params.vocab["<pad>"])]*params.window_size,
                       chars, [tf.constant(params.vocab["<pad>"])]*params.window_size], axis=0),
            tf.concat([[tf.constant(params.vocab["<pad>"])]*params.window_size,
                       tags, [tf.constant(params.vocab["<pad>"])]*params.window_size], axis=0)
        ),
        num_parallel_calls=params.num_threads
    )

    # Convert to dictionary
    dataset = dataset.map(
        lambda line, chars, tags: {
            "origin": line,
            "chars": tf.to_int32(chars),
            "tags": tf.to_int32(tags),
            "start": tf.to_int32(tf.constant(params.window_size)),
            "end": tf.to_int32(tf.shape(chars)[0]-params.window_size)
        },
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(params.batch_size, padded_shapes={"origin":(),"chars": (None,), "tags": (None,),
                                                                     "start": (), "end": ()})

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


def get_validation_input(params):
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
        return line, chars, tags

    dataset = tf.data.Dataset.from_generator(get_generator(params.reference, fn), (tf.string,tf.int32, tf.int32),
                                             ((),(None,), (None,)))
    # Append <pad> symbol
    dataset = dataset.map(
        lambda line, chars, tags:(
            line,
            tf.concat([[tf.constant(params.vocab["<pad>"])]*params.window_size,
                       chars, [tf.constant(params.vocab["<pad>"])]*params.window_size], axis=0),
            tf.concat([[tf.constant(params.vocab["<pad>"])]*params.window_size,
                       tags, [tf.constant(params.vocab["<pad>"])]*params.window_size], axis=0)
        ),
        num_parallel_calls=params.num_threads
    )

    # Convert to dictionary
    dataset = dataset.map(
        lambda line, chars, tags: {
            "origin": line,
            "chars": tf.to_int32(chars),
            "tags": tf.to_int32(tags),
            "start": tf.to_int32(tf.constant(params.window_size)),
            "end": tf.to_int32(tf.shape(chars)[0]-params.window_size)
        },
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(params.batch_size, padded_shapes={"origin":(),"chars": (None,), "tags": (None,),
                                                                     "start": (), "end": ()})

    iterator = dataset.make_initializable_iterator()
    iterator_initializer = iterator.initializer
    features = iterator.get_next()
    return iterator_initializer, features


def get_inference_input(params):
    def fn(line):
        chars = []
        line = line.strip()
        for char in line:
            chars.append(char2id(params.vocab, char))
        return chars, line

    dataset = tf.data.Dataset.from_generator(get_generator(params.input, fn), (tf.int32,tf.string))

    # Append <pad> symbol
    dataset = dataset.map(
        lambda chars, line:
        (tf.concat([[tf.constant(params.vocab["<pad>"])]*params.window_size,
                       chars, [tf.constant(params.vocab["<pad>"])]*params.window_size], axis=0),
         line),
        num_parallel_calls=params.num_threads
    )

    # Convert to dictionary
    dataset = dataset.map(
        lambda chars, line: {
            "chars": tf.to_int32(chars),
            "start": tf.to_int32(tf.constant(params.window_size)),
            "end": tf.to_int32(tf.shape(chars)[0]-params.window_size),
            "origin": line,
        },
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(params.batch_size, padded_shapes={"chars": (None,),
                                                                     "start": (), "end": (),
                                                                     "origin": ()})

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return  features


if __name__ == "__main__":
    pku_train = "/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/training/pku_training.utf8"
    pku_test = "/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/testing/pku_test.utf8"
    pku_gold = "/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/gold/pku_test_gold.utf8"
    params = tf.contrib.training.HParams(
        input = pku_train,
        reference=pku_gold,
        validation=pku_test,
        tag = {"B":0, "M":1, "E":2, "S":3},
        vocab = {"<oov>":1,
                 "<pad>":0},
        num_threads = 6,
        window_size=3,
        batch_size = 2,
        buffer_size = 100,
    )
    res = build_vocab_trans(pku_train, params.tag)
    features = get_trainning_input(params)
    validation_initialize, validation_features = get_validation_input(params)

    params_infer = tf.contrib.training.HParams(
        input = pku_test,
        tag = {"B":0, "M":1, "E":2, "S":3},
        vocab = {"<oov>":1,
                 "<pad>":0},
        num_threads = 6,
        window_size=3,
        batch_size = 2,
        buffer_size = 100,
    )

    features_infer = get_inference_input(params_infer)

    with tf.Session() as sess:
        while True:
            # sess.run(validation_initialize)
            res  = sess.run(features_infer)
            for k, v in res.items():
                print("k", k)
                if k=="origin":
                    print("v", [vv.decode("utf-8") for vv in v])
                else:
                    print("v", v)
            x = input()