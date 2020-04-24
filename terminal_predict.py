# coding=utf-8

"""
@Author:lynn

example:
    python3 terminal_predict.py \
    --model_type bert \
    --labels ./labels.txt \
    --output_dir germeval \
    --max_seq_length  128 \

When console displays 'Input chinese sentence:', then you input some sentences and get pos result.
More detail see ./example/how_to_use.ipynb
"""

import datetime
import collections
import glob
import math
import os
import re

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from transformers import (
    TF2_WEIGHTS_NAME,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoTokenizer,
    TFAutoModelForTokenClassification,
)
from utils_ner import convert_examples_to_features, get_labels, InputExample


MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)



flags.DEFINE_string("model_type", None, "Model type selected in the list: " + ", ".join(MODEL_TYPES))

flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "labels", "", "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."
)

flags.DEFINE_integer(
    "max_seq_length",
    128,
    "The maximum total input sentence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter "
    "will be padded.",
)

flags.DEFINE_string(
    "tpu",
    None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.",
)

flags.DEFINE_integer("num_tpu_cores", 8, "Total number of TPU cores to use.")

flags.DEFINE_boolean("do_lower_case", False, "Set this flag if you are using an uncased model.")

flags.DEFINE_boolean("no_cuda", False, "Avoid using CUDA when available")

# flags.DEFINE_boolean("overwrite_output_dir", False, "Overwrite the content of the output directory")

flags.DEFINE_boolean("overwrite_cache", False, "Overwrite the cached training and evaluation sets")

flags.DEFINE_boolean("fp16", False, "Whether to use 16-bit (mixed) precision instead of 32-bit")

flags.DEFINE_string(
    "gpus",
    "0",
    "Comma separated list of gpus devices. If only one, switch to single "
    "gpu strategy, if None takes all the gpus available.",
)


def read_examples_from_line(line):
    guid_index = 1
    examples = []
    words = []
    labels = []
    for word in line:
        words.append(word)
        labels.append("O")

    if words:
        examples.append(InputExample(
            guid="{}-{}".format('predict', guid_index), words=words, labels=labels)
        )

    return examples

def main(_):
    logging.set_verbosity(logging.INFO)
    args = flags.FLAGS.flag_values_dict()

    if args["fp16"]:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    if args["tpu"]:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args["tpu"])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        args["n_device"] = args["num_tpu_cores"]
    elif len(args["gpus"].split(",")) > 1:
        args["n_device"] = len([f"/gpu:{gpu}" for gpu in args["gpus"].split(",")])
        strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{gpu}" for gpu in args["gpus"].split(",")])
    elif args["no_cuda"]:
        args["n_device"] = 1
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        args["n_device"] = len(args["gpus"].split(","))
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:" + args["gpus"].split(",")[0])

    logging.warning(
        "n_device: %s, distributed training: %s, 16-bits training: %s",
        args["n_device"],
        bool(args["n_device"] > 1),
        args["fp16"],
    )

    labels = get_labels(args["labels"])
    pad_token_label_id = -1

    logging.info("predict parameters %s", args)
    tokenizer = AutoTokenizer.from_pretrained(args["output_dir"], do_lower_case=args["do_lower_case"])
    model = TFAutoModelForTokenClassification.from_pretrained(args["output_dir"])

    while True:
        print('Input chinese sentence:')
        line = str(input())
        if line == 'quit':
            break
        if len(line) < 1:
            print('Please input a chinese sentence or "quit" to break this loop:')
            continue

        examples = read_examples_from_line(line)
        features = convert_examples_to_features(
            examples,
            labels,
            args["max_seq_length"],
            tokenizer,
            cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args["model_type"] in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args["model_type"] in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )
        
        feature = features[0]
        X = collections.OrderedDict()

        X["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.input_ids)))
        X["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.input_mask)))
        X["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.segment_ids)))
        X["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.label_ids)))
        tf_example = tf.train.Example(features=tf.train.Features(feature=X))
        tf_example = tf_example.SerializeToString()

        max_seq_length = args["max_seq_length"]
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "label_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }

        def _decode_record(record):
            example = tf.io.parse_single_example(record, name_to_features)
            features = {}
            features["input_ids"] = example["input_ids"]
            features["input_mask"] = example["input_mask"]
            features["segment_ids"] = example["segment_ids"]
            return features, example["label_ids"]

        dataset = []
        dataset.append(tf_example)

        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.map(_decode_record)

        batch_size = 1
        dataset = dataset.batch(batch_size)
     
        eval_features, eval_labels = iter(dataset).next()
     
        inputs = {"attention_mask": eval_features["input_mask"], "training": False}

        if args["model_type"] != "distilbert":
            inputs["token_type_ids"] = (
                eval_features["segment_ids"] if args["model_type"] in ["bert", "xlnet"] else None
            )

        with strategy.scope():
            logits = model(eval_features["input_ids"], **inputs)[0]
            active_loss = tf.reshape(eval_labels, (-1,)) != pad_token_label_id

        preds = logits.numpy()
        label_ids = eval_labels.numpy()

        preds = np.argmax(preds, axis=2)
        y_pred = [[] for _ in range(label_ids.shape[0])]

        for i in range(label_ids.shape[0]):
            for j in range(label_ids.shape[1]):
                if label_ids[i, j] != pad_token_label_id:
                    y_pred[i].append(labels[preds[i, j]])


        tokens = tokenizer.tokenize(line)
        print('## tokens = %s' % tokens)
        print('## y_pred = %s' % y_pred)
        print('## %s = %s' % (len(tokens), len(y_pred[0])))
        word_group = []
        subword = {}
        
        def _add_word(subword):
            word_group.append(subword['token'] + '/' + subword['flag'])
            subword.clear()

        for i, token in enumerate(tokens):
            flag = y_pred[0][i]
            print('## %s = %s' % (token, flag))
            if flag.startswith('B'):
                if len(subword) > 0:
                    _add_word(subword)
                subword['token'] = token
                subword['flag'] = flag
            elif flag.startswith('I'):
                if (
                    len(subword) > 0
                    and (y_pred[0][i-1].startswith('I') or y_pred[0][i-1].startswith('B'))
                    and (y_pred[0][i-1][1:] == flag[1:])
                ):
                    subword['token'] = subword['token'] + token
                    continue
                elif len(subword) > 0:
                    _add_word(subword)
                subword['token'] = token
                subword['flag'] = flag
            else:
                if len(subword) > 0:
                    _add_word(subword)
                subword['token'] = token
                subword['flag'] = flag
                _add_word(subword)

        if len(subword) > 0:
            _add_word(subword)
        print('## word_group = %s' % word_group)


if __name__ == "__main__":
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_type")
    app.run(main)