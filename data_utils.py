# coding=utf-8
import collections
import os
import tensorflow as tf
from absl import app, flags, logging
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file

def load_cache(cached_file, max_seq_length):
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

    d = tf.data.TFRecordDataset(cached_file)
    d = d.map(_decode_record, num_parallel_calls=4)
    count = d.reduce(0, lambda x, _: x + 1)

    return d, count.numpy()

def save_cache(features, cached_features_file):
    writer = tf.io.TFRecordWriter(cached_features_file)

    for (ex_index, feature) in enumerate(features):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(features)))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        record_feature = collections.OrderedDict()
        record_feature["input_ids"] = create_int_feature(feature.input_ids)
        record_feature["input_mask"] = create_int_feature(feature.input_mask)
        record_feature["segment_ids"] = create_int_feature(feature.segment_ids)
        record_feature["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=record_feature))

        writer.write(tf_example.SerializeToString())

    writer.close()

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, batch_size, mode):
    drop_remainder = True if args["tpu"] or mode == "train" else False

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args["data_dir"],
        "cached_{}_{}_{}.tf_record".format(
            mode, list(filter(None, args["model_name_or_path"].split("/"))).pop(), str(args["max_seq_length"])
        ),
    )
    if os.path.exists(cached_features_file) and not args["overwrite_cache"]:
        logging.info("Loading features from cached file %s", cached_features_file)
        dataset, size = load_cache(cached_features_file, args["max_seq_length"])
    else:
        logging.info("Creating features from dataset file at %s", args["data_dir"])
        examples = read_examples_from_file(args["data_dir"], mode)
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
        logging.info("Saving features into cached file %s", cached_features_file)
        save_cache(features, cached_features_file)
        dataset, size = load_cache(cached_features_file, args["max_seq_length"])

    if mode == "train":
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=8192, seed=args["seed"])

    dataset = dataset.batch(batch_size, drop_remainder)
    dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset, size