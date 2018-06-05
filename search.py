import numpy as np
import data


def recover_line(param, line, start, end, logits):
    transition = param.transition
    tag_type = len(param.tag)

    # for i in range(start, end):

    data.id2char()
    newline = None
    return newline


def recover_features(params, features):
    newlines = []
    assert params.batch_size == len(features["chars"])
    for i in range(params.batch_size):
        newlines.append(recover_line(params, features["chars"][i], features["start"][i],
                                     features["end"][i], features["logits"][i]))
    return newlines