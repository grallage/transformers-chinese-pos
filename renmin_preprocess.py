# coding=utf-8

"""
@Author:lynn

This's a simple corpus preprocess script. We download People's Daily data from 
https://raw.githubusercontent.com/InsaneLife/ChineseNLPCorpus/master/NER/renMinRiBao/renmin.txt
and run this command in console:
    python preprocess ./renmin.txt
then we get train.txt, dev.txt and test.txt in renmin.txt document.

Note: this scripy only suppory renmin.txt data because it has special data format.
"""

import sys
import os
import unicodedata

dataset = sys.argv[1]
max_len = int(sys.argv[2])

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def format_each_line(line):
    """tranform raw line to special format.
    for example, some sentences like this:
        [中国/ns  国际/n  广播/vn  电台/n]nt  和/c  [中央/n  电视台/n]nt 
    We need to transform to:
        ['中国国际广播电台/nt', '和/c', '中央电视台/nt']
    """
    word_pair_flag = False
    index_start = 0
    result = []
    for index_end in range(len(line) - 1): # each line contain space character in the end
        word = line[index_end]
        if word == ' ' and line[index_end + 1] == ' ':            
            if word_pair_flag:
                continue
            token = line[index_start:index_end].strip()
            if token.startswith('['):
                token_ = ''
                pair_words = token.split('[')[1].split(']')[0].split('  ')
                for pair_word in pair_words:
                    token_ += pair_word.split('/')[0]
                token_ = token_ + '/' + token[token.rfind(']') + 1:]
                result.append(token_)
            else:
                result.append(token)
            index_start = index_end + 1
        elif word == '[':
            word_pair_flag = True
        elif word == ']':
            word_pair_flag = False

    return result

def get_token_and_flag(line):
    """get tokens and flags in each line
    input: 
        ['中国国际广播电台/nt', '和/c', '中央电视台/nt']
    output:
        (['中国国际广播电台', '和', '中央电视台']
        ['NT', 'C', 'NT'])
    """
    tokens=[]
    flags=[]
    words_length = 0
    for index, pairStr in enumerate(line):
        if index == 0:
            continue
        i = pairStr.rfind('/')
        token = pairStr[:i]
        words_length += len(token)
        tokens.append(token)
        flags.append(pairStr[i + 1:].upper())
    return tokens, flags, words_length

def split_sentence(tokens, flags):
    words_length = 0
    last_symbol_index = 0
    start_index = 0
    result = []
    for i, token in enumerate(tokens):
        if token in '。、，；' and words_length + len(token) < max_len:
            last_symbol_index = i
            words_length += len(token)
        
        elif words_length + len(token) > max_len:
            if start_index != last_symbol_index:                
                result.append((tokens[start_index:last_symbol_index], flags[start_index:last_symbol_index]))
                start_index = last_symbol_index
                words_length = 0
                for t in tokens[last_symbol_index:i]:
                    words_length += len(t)
            else:
                result.append((tokens[start_index:i], flags[start_index:i]))
                start_index = i
                words_length = len(token)
        else:
            words_length += len(token)

    return result

## start preprocessing
corpus = []
with open(dataset, "r") as f:    
    for line in f:        
        line = format_each_line(line)        
        tokens, flags, words_length = get_token_and_flag(line)
        tf_pairs = []
        if words_length > max_len:
            tf_pairs.extend(split_sentence(tokens, flags))
        else:
            tf_pairs.append((tokens, flags))
        for tokens, flags in tf_pairs:
            assert len(tokens) == len(flags)
            new_line = []
            for i in range(len(tokens)):
                new_line.append(tokens[i] + " " + flags[i])
            corpus.append(new_line)

parent_path = os.path.dirname(dataset)
train_f = os.path.join(parent_path, 'train.txt')
test_f = os.path.join(parent_path, 'test.txt')
dev_f = os.path.join(parent_path, 'dev.txt')
labels_f = os.path.join(parent_path, 'labels.txt')
label_set = set()
corpus_size = len(corpus)
train_size = int(corpus_size * 0.8)
dev_size = (corpus_size-train_size) // 2
print('corpus_size = %s, train_size = %s, dev_size = %s' % (corpus_size,train_size,dev_size))

with open(train_f, 'w') as f1, open(dev_f, 'w') as f2,\
    open(test_f, 'w') as f3, open(labels_f, 'w') as f4:

    
    for i, token_pairs in enumerate(corpus):
        if i < train_size:
            f = f1
        elif i < train_size + dev_size:
            f = f2
        else:
            f = f3
        write_blank = False
        
        for line in token_pairs:
            if len(remove_control_characters(line)) == 0:
                continue

            write_blank = True
            index = line.rfind(' ')
            words = line[:index]
            flag = line[index+1:]

            if len(words) > 1:
                for j in range(len(words)):
                    word = words[j]
                    if j == 0:
                        new_flag = 'B-' + flag
                    else:
                        new_flag = 'I-' + flag
                    label_set.add(new_flag)
                    f.write(word + ' ' + new_flag + '\n')
            else:
                new_flag = 'O-' + flag
                label_set.add(new_flag)
                f.write(words + ' ' + new_flag + '\n')

        if (write_blank):            
            f.write('\n')

    for label in sorted(label_set):
        f4.write(label + '\n')
