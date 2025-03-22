# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import sys
import cv2
import math
import json
import torch
import numpy as np
from typing import Iterable
from tqdm import tqdm
import time
import util.misc as utils

def get_loss(model, device,samples, input_label_seqs, prompts, text_length, criterion, output_seqs):
    outputs_pre = model(samples, input_label_seqs, prompts, text_length)
    seq_loss = criterion(outputs_pre.transpose(1, 2), output_seqs)
    return seq_loss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    lr_scheduler: list = [0], print_freq: int = 10, text_length: int = 25, args=None):
    start_time = time.time()
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    optimizer.param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer.param_groups[1]['lr'] = lr_scheduler[epoch] * 0.1

    for samples, input_label_seqs, output_label_seqs, prompts in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        input_label_seqs = input_label_seqs.to(device)
        output_label_seqs = output_label_seqs.to(device)
        if not all(input_label_seqs.tolist()):
            continue

        output_seqs = output_label_seqs
        seq_loss = get_loss(model, device, samples, input_label_seqs, prompts, text_length, criterion, output_seqs)
        loss_dict = {'seq_loss':seq_loss}
        weight_dict = {'seq_loss':1}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
         
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print('Epoch {} training time: {:.2f}min'.format(epoch, (time.time() - start_time)/60))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, model, criterion, data_loader, device, output_dir, chars, start_index, visualize=False, text_length=25):
    model.eval()
    criterion.eval()
    chars = list(chars)
    import time
    cnt = 0
    total = 0
    results = []
    for samples, targets, prompt in tqdm(data_loader):
        batch = len(targets)
        samples = samples.to(device)
        dataset_names = [target['dataset_name'] for target in targets]
        prompt = prompt
        seq = torch.ones(1, 1, dtype=torch.long).to(device) * start_index
        torch.cuda.synchronize()
        t0 = time.time()
        output, prob = model(samples, seq,prompt, text_length)
        torch.cuda.synchronize()
        t1 = time.time()
        cnt += 1
        total += t1-t0
        output = output[0]
        prob = prob[0]
        if args.test_type == "vqa":
            result = decode_pred_seq_vqa(output, prob, samples, targets[0], args)
        else:
            result = decode_pred_seq(output, prob, samples, targets[0], args)
        results.extend(result)
        if args.visualize:
            image = cv2.imread(os.path.join(targets[0]['image_folder'], targets[0]['file_name']))
            image = visualize_decoded_result(image, result)
            save_path = os.path.join(output_dir, 'vis', targets[0]['file_name'])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

    json_path = os.path.join(output_dir, 'results', dataset_names[0]+'.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    results_json = json.dumps(results, indent=4)
    with open(json_path, 'w') as f:
        f.write(results_json)    

def decode_pred_seq(index_seq, prob_seq, samples, meta_info, args):
    index_seq = index_seq[:-1]
    prob_seq = prob_seq[:-1]
    if args.train_point =="quad":
        if len(index_seq) % (args.max_length+8) != 0:
            index_seq = index_seq[:-len(index_seq)%(args.max_length+8)]
            prob_seq = index_seq[:-len(index_seq)%(args.max_length+8)]
    else:
        if len(index_seq) % (args.max_length+2) != 0:
            index_seq = index_seq[:-len(index_seq)%(args.max_length+2)]
            prob_seq = index_seq[:-len(index_seq)%(args.max_length+2)]
    
    decode_results = decode_seq(index_seq, 'none', args)
    if args.train_point =="quad":
        confs = prob_seq.reshape(-1, (args.max_length+8)).mean(-1)
    else:
        confs = prob_seq.reshape(-1, (args.max_length+2)).mean(-1)
    
    image_h, image_w = meta_info['orig_size']
    results = []
    print(int(meta_info['file_name'].split('_')[-1].split('.')[0]))
    for decode_result, conf in zip(decode_results, confs):
        points = decode_result['point']
        polys = [[points[i]*image_w, points[i+1]*image_h] for i in range(0, len(points), 2)]
        polys = [[point[0].item(), point[1].item()] for point in polys]
        recog = decode_result['recog']
        result = {
            'image_id': int(meta_info['file_name'].split('_')[-1].split('.')[0]),
            'filename': meta_info['file_name'],
            'image_folder': str(meta_info['image_folder']),
            'polys': polys,
            'rec': recog,
            'score': conf.item()
        }
        print(polys,recog)
        results.append(result)
    
    return results

def decode_pred_seq_vqa(index_seq, prob_seq, samples, meta_info, args):
    index_seq = index_seq[:-1]
    prob_seq = prob_seq[:-1]
    decode_results = decode_seq_vqa(index_seq, 'none', args)
    results = []
    for decode_result in decode_results:
        recog = decode_result['recog']
        result = {
            'answer': recog,
            'question_id': int(meta_info['question_id'])
        }
        results.append(result)
    return results

def decode_seq(seq, type, args):
    seq = seq[seq != (args.padding_index)]
    if args.train_point =="quad":
        seq = seq.reshape(-1, (args.max_length + 8))
        coor_length = 8
    else:
        seq = seq.reshape(-1, (args.max_length + 2))
        coor_length = 2

    decode_result = []
    for text_ins_seq in seq:
        point = []
        for index in text_ins_seq[:coor_length]:
            point_xy = index / args.bins
            point.append(point_xy)
        recog = []
        for index in text_ins_seq[coor_length:]:
            if index == (args.pad_rec_index):
                break
            if index == (args.pad_rec_index) - 1:
                continue
            
            recog.append(args.chars[index - args.category_start_index_text])

        recog = ''.join(recog)
        decode_result.append(
            {'point': point, 'recog': recog}
        )

    return decode_result

def decode_seq_vqa(seq, type, args):
    seq = seq[seq != (args.padding_index)]
    decode_result = []
    recog = str()
    recog_word = []
    for index in seq:
        if index == (args.pad_rec_index):
            break
        if index == (args.pad_rec_index) - 1:
            continue
        recog_word.append(args.chars[index - args.category_start_index_text])
    recog_word = ''.join(recog_word)
    print(recog_word)
    recog = recog_word
    decode_result.append(
        {'recog': recog}
    )
    return decode_result

def visualize_decoded_result(image, results):
    for polys in results:
        poly = polys['polys']
        if len(poly)==4:
            poly = np.array(poly).reshape(4, 2)
            poly = poly.astype(np.int32)
            cv2.polylines(image, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
        else:
            center = (int(poly[0][0]), int(poly[0][1]))
            center= np.array(center).astype(np.int32)
            cv2.circle(image, center, radius=5, color=(255, 0, 0), thickness=-1)
    return image