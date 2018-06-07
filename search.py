import math
INF = 1e30


def recover_line_viterbi(param, line, start, end, logprobs):
    transition = param.transition
    transition = [[prob/sum(line) for prob in line] for line in transition]
    logtransition = [[-INF if prob==0.0 else math.log(prob) for prob in line] for line in transition]

    tag_type = len(param.tag)

    dp = [[0 for i in range(tag_type)] for j in range(len(logprobs))]
    mark = [[0 for i in range(tag_type)] for j in range(len(logprobs))]

    for i in range(start, end):
        if i==start:
            for j in range(tag_type):
                dp[i][j]=logprobs[i][j]
                mark[i][j]=-1
        else:
            for j in range(tag_type):
                max_index=0
                max_value=dp[i-1][0]+logtransition[0][j]
                for k in range(tag_type):
                    if dp[i-1][k]+logtransition[k][j]>max_value:
                        max_value=dp[i-1][k]+logtransition[k][j]
                        max_index=k
                dp[i][j] = max_value
                mark[i][j] = max_index

    # find the biggest value
    max_index = 0
    max_value = dp[end-1][0]
    for i in range(tag_type):
        if dp[end-1][i] > max_value:
            max_value = dp[end-1][i]
            max_index = i

    # trace back
    track_back_tag = [max_index]
    for i in range(end-1, start, -1):
        track_back_tag.append(mark[i][track_back_tag[-1]])

    reversed(track_back_tag)


    line = line.decode("utf-8").strip()
    newline = ""
    assert len(line)==len(track_back_tag),"tags numgber is different from chars number"
    for i in range(len(track_back_tag)):
        if track_back_tag[i] == param.tag["B"] or \
                track_back_tag[i] == param.tag["M"]:
            newline += line[i]
        elif track_back_tag[i] == param.tag["E"] or \
                track_back_tag[i] == param.tag["S"]:
            newline += line[i]
            if i != end-1:
                newline += " "
            else:
                newline += ""


    # debug
    # print("=====debug=====")
    # print("line", line)
    # print("start", start)
    # print("end", end)
    # print("logprob", logprobs)
    # print("logtransition", logtransition)
    # print("dp", dp)
    # print("mark", mark)
    # print("track_back_tag", track_back_tag)
    # x = input()

    return newline


def recover_line_greedy(param, line, start, end, logprobs):
    tag = []
    line = line.decode("utf-8")

    def find_max(listlike):
        index=-1
        value=-1
        for i in range(len(listlike)):
            if listlike[i]>value:
                index=i
                value=listlike[i]
        return index, value

    for i in range(start, end):
        temp_tag, _ = find_max(logprobs[i])
        tag.append(temp_tag)

    newline = ""
    for i in range(len(tag)):
        if tag[i] == param.tag["B"] or \
                tag[i] == param.tag["M"]:
            newline += line[i]
        elif tag[i] == param.tag["E"] or \
                tag[i] == param.tag["S"]:
            newline += line[i]
            if i != end-1:
                newline += " "
            else:
                newline += ""
    return newline


def recover_features(params, features):
    newlines = []
    assert params.batch_size == len(features["chars"])

    if params.search_policy == "viterbi":
        recover_line = recover_line_viterbi
    elif params.search_policy == "greedy":
        recover_line = recover_line_greedy
    else:
        raise NotImplementedError(params.search_policy+"is not implemeted")

    for i in range(params.batch_size):
        newlines.append(recover_line(params, features["origin"][i], features["start"][i],
                                     features["end"][i], features["logprobs"][i]))
    return newlines