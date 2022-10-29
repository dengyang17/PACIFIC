import re
import os
import argparse
import json
from tatqa_metric import *
from sklearn.metrics import f1_score, precision_score, recall_score
import pyrouge
import logging
from collections import Counter
import string


def rouge(reference, candidate, log_path):
    """
    compute the rouge score
    :param reference: reference
    :param candidate: candidate
    :param log_path: path to log
    :return: rouge-2 score
    """
    # check if of equal amount.
    assert len(reference) == len(candidate)
    # directory for saving sentences
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    # check if there are directories for reference and candidate
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    # write files
    for i in range(len(reference)):
        with open(ref_dir+"%06d_reference.txt" % i, 'w', encoding='utf-8') as f:
            f.write(reference[i] + '\n')
        with open(cand_dir+"%06d_candidate.txt" % i, 'w', encoding='utf-8') as f:
            f.write(candidate[i] + '\n')

    # use pyrouge and ROUGE155
    r = pyrouge.Rouge155(log_level=logging.CRITICAL)
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    # compute the scores
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    # recall
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    # precision
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    # f score
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]

    return f_score[:], recall[:], precision[:]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def qr_f1_em(targets, preds):
    f1 = exact_match = total = 0
    for t, p in zip(targets, preds):
        total += 1
        if normalize_answer(t) == normalize_answer(p):
            exact_match += 1
        prediction_tokens = normalize_answer(p).split()
        ground_truth_tokens = normalize_answer(t).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 += 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 += (2 * precision * recall) / (precision + recall)
    return [float(f1)/total, float(exact_match)/total]


def calculate(raw_pred, raw_ref, data_name, set_name, model_name):
    print('Dataset: ', data_name, '-----------------')

    em_and_f1 = TaTQAEmAndF1()
    with open('../data/{}/{}.json'.format(data_name, set_name), 'r', encoding='utf-8') as infile:
        raw_ref = json.load(infile)

    gold_answers = []
    gold_goals = []
    uids = []
    pred_json = {}
    for qas in raw_ref:
        for qa in qas["questions"]:
            gold_answers.append(qa)
            gold_goals.append(int(qa['req_clari']))
            uids.append(qa['uid'])
    
    assert len(gold_answers) == len(raw_pred)

    pred_goals = []
    for pred, ref, uid in zip(raw_pred, gold_answers, uids):
        pred_answer, pred_scale, pred_derivation, pred_goal = process_answer(pred, model_name)
        pred_goals.append(pred_goal)
        pred_json[uid] = [pred_answer, pred_scale]
        #if ref['answer'] != pred_answer:
        #    print(ref['answer'], pred_derivation, pred_answer)
        em_and_f1(ground_truth=ref, prediction=pred_answer, pred_scale=pred_scale)
    
    with open('output/{}/{}_{}.json'.format(data_name,data_name,set_name), 'w', encoding='utf-8') as outfile:
        json.dump(pred_json, outfile, indent=2)

    # CNP evaluation
    precision = precision_score(gold_goals, pred_goals, average='macro', zero_division=0)
    recall = recall_score(gold_goals, pred_goals, average='macro', zero_division=0)
    f1 = f1_score(gold_goals, pred_goals, average='macro', zero_division=0)
    auto_scores = [precision, recall, f1]


    global_em, global_f1, global_scale, global_op = em_and_f1.get_overall_metric()
    print("----")
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("Scale score {0:.2f}".format(global_scale * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")

    detail_raw = em_and_f1.get_raw_pivot_table()
    print("---- raw detail ---")
    print(detail_raw)
    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    print("---- em detail ---")
    print(detail_em)
    print("---- f1 detail ---")
    print(detail_f1)  
    return {'cqa': [global_em, global_f1], 'cnp': auto_scores}


def calculate_beam(raw_pred, raw_ref, data_name, set_name, model_name):
    print('Dataset: ', data_name, '-----------------')

    em_and_f1 = TaTQAEmAndF1()
    with open('../data/{}/{}.json'.format(data_name, set_name), 'r', encoding='utf-8') as infile:
        raw_ref = json.load(infile)

    gold_answers = []
    gold_goals = []
    uids = []
    pred_json = {}
    for qas in raw_ref:
        for qa in qas["questions"]:
            gold_answers.append(qa)
            gold_goals.append(qa['req_clari'])
            uids.append(qa['uid'])
    
    assert len(gold_answers) == len(raw_pred)
    
    pred_goals = []
    for pred, ref, uid in zip(raw_pred, gold_answers, uids):
        answer_list = []
        answer_dict = {}
        for b in pred:
            pred_answer, pred_scale, pred_derivation, pred_goal = process_answer(b, model_name)
            answer_list.append((str(pred_answer), pred_scale))
            if (str(pred_answer), pred_scale) not in answer_dict:
                answer_dict[(str(pred_answer), pred_scale)] = []
            answer_dict[(str(pred_answer), pred_scale)].append((pred_answer, pred_scale, pred_derivation, pred_goal))
        count = Counter(answer_list)
        final_pred = count.most_common()[0][0]
        #if ref['answer'] != pred_answer:
        #    print(ref['answer'], count.items())
        #print(answer_dict[final_pred])
        pred_goals.append(answer_dict[final_pred][0][3])
        pred_json[uid] = [pred_answer, pred_scale]
        em_and_f1(ground_truth=ref, prediction=answer_dict[final_pred][0][0], pred_scale=answer_dict[final_pred][0][1])
    
    with open('output/{}/{}_{}_cv.json'.format(data_name,data_name,set_name), 'w', encoding='utf-8') as outfile:
        json.dump(pred_json, outfile, indent=2)

    # CNP evaluation
    precision = precision_score(gold_goals, pred_goals, average='macro', zero_division=0)
    recall = recall_score(gold_goals, pred_goals, average='macro', zero_division=0)
    f1 = f1_score(gold_goals, pred_goals, average='macro', zero_division=0)
    auto_scores = [precision, recall, f1]


    global_em, global_f1, global_scale, global_op = em_and_f1.get_overall_metric()
    print("----")
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("Scale score {0:.2f}".format(global_scale * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")

    detail_raw = em_and_f1.get_raw_pivot_table()
    print("---- raw detail ---")
    print(detail_raw)
    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    print("---- em detail ---")
    print(detail_em)
    print("---- f1 detail ---")
    print(detail_f1)  
    return {'cqa': [global_em, global_f1], 'cnp': auto_scores}




def process_answer(pred, model_name):
    if model_name.startswith('codet5'):
        tmp_goal = re.findall(r"\[goal\] (.+?)<s>", pred)
    else:
        tmp_goal = re.findall(r"\[goal\] (.+?) \[system\]", pred)
    if len(tmp_goal) > 0:
        try:
            pred_goal = int(eval(tmp_goal[0]))
        except Exception as e:
            pred_goal = 0
    else:
        pred_goal = 0

    if '[system]' not in pred:
        return None, None, None, pred_goal

    if '[scale]' in pred:
        pred = pred.split('[scale]')
        if '[system]' not in pred[0]:
            return None, None, None, pred_goal
        pred_derivation = pred[0].split('[system]')[1].strip()
        pred_scale = pred[1].split('</s>')[0].strip()
    else:
        pred = pred.split('[system]')[1]
        pred_derivation = pred.split('</s>')[0].strip()
        pred_scale = ''
            
    try:
        answer = eval(pred_derivation)
        if type(answer) == tuple:
            tmp = re.sub(r'[,$%a-zA-Z]+', '', pred_derivation)
            answer = eval(tmp)
    except Exception as e:
        try:
            if pred_derivation.startswith("[\'") and pred_derivation.endswith("\']"):
                tmp = pred_derivation[2:-2]
                tmp = "[\"" + tmp + "\"]"
                answer = [eval(tmp)]
            else:
                tmp = re.sub(r'[,$%a-zA-Z]+', '', pred_derivation)
                answer = eval(tmp)
        except Exception as e:
            answer = pred_derivation
    if type(answer) == float:
        if answer < 1:
            answer = round(answer*100, 2)
        else:
            answer = round(answer, 2)
    pred_answer = answer
    
    return pred_answer, pred_scale, pred_derivation, pred_goal

if __name__ == '__main__':
    data = 'pacific'
    set_name = 'validation'
    model_name = 'codet5-base'
    pred = []
    with open('output/{}/{}_{}_{}.decoded'.format(data,data,set_name, model_name), 'r', encoding='utf-8') as infile:
        for line in infile:
            pred.append(line.strip())
    with open('../data/{}/{}.json'.format(data,set_name), 'r', encoding='utf-8') as infile:
        ref = json.load(infile)
    sample_pred = []
    with open('output/{}/{}_{}_{}_beam.decoded'.format(data,data,set_name, model_name), 'r', encoding='utf-8') as infile:
        for line in infile:
            sample_pred.append(eval(line.strip()))

    auto_metric = calculate(pred, ref, data, set_name, model_name)
    auto_metric_beam = calculate_beam(sample_pred, ref, data, set_name, model_name)
    print(auto_metric)
    print(auto_metric_beam)
