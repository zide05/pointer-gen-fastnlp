import argparse
import torch
import sys
import os
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.stem import WordNetLemmatizer
import json

# stopwords = stopwords.words('english')
# stop_list = [',', '.', '?', '!', '\'', '`', '\'s', ':', '-', '-lrb-', '-rrb-', '#', '--', '\'\'', '``', 'n\'t', '$',
#              "(", ")", "@", "+", "[", "]", "*", '&']
# for w in stop_list:
#     stopwords.append(w)

stopwords = stopwords.words('english')
stop_list = [',', '.', '?', '!', '\'', '`', '\'s', ':', '-', '-lrb-', '-rrb-', '#', '--', '\'\'', '``', 'n\'t', '$',
             '(', ')']
for w in stop_list:
    stopwords.append(w)


def get_ngram(sentences, n):
    dic = {}
    if n == -1:
        for sent in sentences:
            if sent in dic:
                dic[sent] += 1
            else:
                dic[sent] = 1
        return dic

    for sent in sentences:
        tokens = word_tokenize(sent)
        # if n == 1:
        tokens = [token for token in tokens if token not in stopwords]
        s_len = len(tokens)
        for k in range(s_len):
            if k + n > s_len:
                break
            update_key = ' '.join(tokens[k: k + n])
            if update_key in dic:
                dic[update_key] += 1
            else:
                dic[update_key] = 1
    return dic


def repetition(result_path, ngram):
    rate = []
    cnt = 0
    with open(result_path, 'r') as f:
        for line in f:
            cnt += 1
            line = line.strip()
            pred = sent_tokenize(line)
            # pred = line.split("\n")
            dic = get_ngram(pred, ngram)
            cnt_all = 0
            cnt_rep = 0
            for key, value in dic.items():
                cnt_all += value
                if value > 2:
                    cnt_rep += value - 1
            if not cnt_all == 0:
                rate.append(cnt_rep / cnt_all)
    return sum(rate) / len(rate)


def novelty(pred_path, ngram, raw_path):
    raws = []
    preds = []
    with open(raw_path, 'r') as f:
        for line in f:
            line = line.strip()
            raws.append(json.loads(line)["text"])
    with open(pred_path, 'r') as f:
        for line in f:
            line = line.strip()
            preds.append(sent_tokenize(line))
    assert len(preds) == len(raws), "preds and raws have inequal number of samples!"
    rate = []
    for i in range(len(preds)):
        score = 0
        all_cnt = 0
        pred_dic = get_ngram(preds[i], ngram)
        raw_keys = get_ngram(raws[i], ngram).keys()
        for key, value in pred_dic.items():
            all_cnt += value
            if not key in raw_keys:
                score += value
        if not all_cnt == 0:
            rate.append(score / all_cnt)
    return sum(rate) / len(rate)


def novelty1(pred_path, ngram, raw_path):
    raws = []
    preds = []
    with open(raw_path, 'r') as f:
        for line in f:
            line = line.strip()
            raws.append(json.loads(line)["text"])
    with open(pred_path, 'r') as f:
        for line in f:
            line = line.strip()
            preds.append(sent_tokenize(line))
    assert len(preds) == len(raws), "preds and raws have inequal number of samples!"
    rate = []
    for i in range(len(preds)):
        score = 0
        all_cnt = 0
        pred_dic = get_ngram(preds[i], ngram)
        raw_keys = get_ngram(sent_tokenize(" ".join(raws[i])), ngram).keys()
        for key, value in pred_dic.items():
            all_cnt += value
            if not key in raw_keys:
                score += value
        rate.append(score / all_cnt)
    return sum(rate) / len(rate)


# python novelty_repetition.py -pred_path ~/tasks/fastnlp-relevant/summarization/my-pnt-sum/log/Xsum/train_pointer_gen_coverage/decode_xsum_best_Model_rouge-l-f_2019-12-31-04-58-33/pred.txt -raw_path ~/Datasets/XSum/finish/xsum.test.jsonl -n_grams 1,2,3,4,-1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred_path", type=str, required=True)
    parser.add_argument("-raw_path", type=str, required=True)
    parser.add_argument("-n_grams", type=str, required=True, help="-1 代表以sentence作为单元")
    parser.add_argument("-save_path", type=str, default="result")
    args = parser.parse_args()

    pred_path = args.pred_path
    raw_path = args.raw_path
    n_grams = [int(tmp.strip()) for tmp in args.n_grams.split(",")]

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    rep_path = os.path.join(args.save_path, "repetition.jsonl")
    novel_path = os.path.join(args.save_path, "novelty.jsonl")

    for n_gram in n_grams:
        print("{} gram ----------".format(n_gram))
        with open(rep_path, 'a') as f:
            f.write(json.dumps({"path": pred_path, "n_gram": n_gram, "repetition_rate": repetition(pred_path, n_gram)}))
            f.write("\n")
        with open(novel_path, 'a') as f:
            f.write(json.dumps({"pred_path": pred_path, "raw_path": raw_path, "n_gram": n_gram,
                                "novelty": novelty(pred_path, n_gram, raw_path)}))
            f.write("\n")
