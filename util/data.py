# Copyright (2023) Bytedance Ltd. and/or its affiliates

def process_args(args):
    args.category_start_index_cls = args.bins + args.padding_bins*2

    num_char_classes = len(args.chars)
    args.num_char_classes=num_char_classes

    args.no_known_char = args.category_start_index_cls + 95
    args.pad_rec_index = args.no_known_char + 1
    args.end_index = args.pad_rec_index+1
    args.start_index = args.end_index + 1
    args.noise_index = args.start_index + 1
    args.padding_index = args.noise_index + 1
    args.no_known_index = args.padding_index
    args.category_start_index_text=args.no_known_index+1
    args.num_classes=args.no_known_index+num_char_classes+1
    return args