"""
This version of baseline is from lsgn-pytorch/src/baseline-re-check-w-mask-full-embedding
Mdify:
    adjust cnn into char emb size: 100, output channel size: 100
    adjust the span representation, not using the word embeddings as the head embeddings
    try stable
    sum the lstm output for the span representation
    Finally, I want to write the unified code for both span-based and word-based SRL...@kiro
    "/home/kiro/Work/SRL/SRL-w-Heterogenous-Dep/src/baseline-for-both-span-and-word-based-SRL"
    baseline add heterogeneous GCNs, copy code from
    Chinese-SRL/src/baseline-with-GCN-4.0-gcn-on-bottom
    + probabilities of heads
    + based on GCN 2.0, which is depend on p
    + hete deps
"""
from neural_srl.shared import *
from neural_srl.shared.tagger_data import TaggerData, get_hete_dep_trees_info
from neural_srl.shared.measurements import Timer
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
from neural_srl.shared.evaluation import SRLEvaluator
from neural_srl.shared.inference_utils import *
from neural_srl.shared.srl_eval_utils import *
from neural_srl.shared.syntactic_extraction import SyntacticCONLL

import argparse
import time
import numpy
import os
import shutil
import torch
# -*- coding: utf-8 -*-


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = "cpu"


def adjust_learning_rate(optimizer, last_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = last_lr * 0.999
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def evaluate_tagger(model, batched_dev_data, hete_deps, eval_data, label_dict, config, evaluator, writer, global_step):
    print "Evaluating"
    torch.cuda.empty_cache()
    dev_loss = 0
    srl_predictions = []

    model.eval()
    for i, batched_tensor in enumerate(batched_dev_data):
        sent_ids, sent_lengths, \
        word_indexes, head_indexes, char_indexes, \
        predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens, \
        gold_predicates, num_gold_predicates = batched_tensor
        hete_dep_trees = get_hete_dep_trees_info(hete_deps, sent_ids, sent_lengths, training=False)

        if args.gpu:
            word_indexes, head_indexes, char_indexes, \
            predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens = \
                word_indexes.cuda(), head_indexes.cuda(), char_indexes.cuda(), predicate_indexes.cuda(), arg_starts.cuda(), \
                arg_ends.cuda(), arg_labels.cuda(), srl_lens.cuda()  # , gold_predicates.cuda(), num_gold_predicates.cuda()

        predicated_dict, loss = model.forward(sent_lengths, word_indexes, head_indexes, char_indexes,
                                              (predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens),
                                              (gold_predicates, num_gold_predicates),
                                              tree_gru_input=hete_dep_trees)
        dev_loss += float(loss)

        decoded_predictions = srl_decode(sent_lengths, predicated_dict, label_dict.idx2str, config)

        if "srl" in decoded_predictions:
            srl_predictions.extend(decoded_predictions["srl"])

    print ('Dev loss={:.6f}'.format(dev_loss / len(batched_dev_data)))

    sentences, gold_srl, gold_ner = zip(*eval_data)
    # Summarize results, evaluate entire dev set.
    precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, srl_label_mat, comp =\
        (compute_srl_f1(sentences, gold_srl, srl_predictions, args.gold, config.span_based))

    if conll_f1 > evaluator.best_accuracy:
        evaluator.best_accuracy = conll_f1
        evaluator.has_best = True
    else:
        evaluator.has_best = False
    writer.write('{}\t{}\t{:.6f}\t{:.2f}\t{:.2f}\n'.format(global_step,
                                                           time.strftime("%Y-%m-%d %H:%M:%S"),
                                                           float(dev_loss),
                                                           float(conll_f1),
                                                           float(evaluator.best_accuracy)))
    writer.flush()
    torch.cuda.empty_cache()
    if evaluator.has_best:
        model.save(os.path.join(args.model, 'model'))


def train_tagger(args):
    # get the parse configuration
    config = configuration.get_config(args.config)
    config.span_based = args.span == "span"
    # set random seeds of numpy and torch
    numpy.random.seed(666)
    torch.manual_seed(666)
    # set pytorch print precision
    torch.set_printoptions(precision=20)
    # set the default number of threads
    torch.set_num_threads(4)
    # GPU of pytorch
    gpu = torch.cuda.is_available()
    if args.gpu and gpu:
        print("GPU available? {}\t and GPU ID is : {}".format(gpu, args.gpu))
        # set pytorch.cuda's random seed
        torch.cuda.manual_seed(666)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with Timer('Data loading'):
        vocab_path = args.vocab if args.vocab != '' else None
        label_path = args.labels if args.labels != '' else None
        gold_props_path = args.gold if args.gold != '' else None

        print ('Task is : {}'.format(args.task))
        assert args.task == 'SRL'
        # Data for SRL.
        data = TaggerData(config, *reader.get_srl_data(config, args.train, args.dep_trees, args.dev,
                                                       vocab_path, label_path))
        # Generate SRL evaluator for Dev data
        """Actually, this evaluator has been abandoned, and the only function is to store the highest accuracy."""
        evaluator = SRLEvaluator(data.get_development_data(),
                                 data.label_dict,
                                 gold_props_file=gold_props_path,
                                 pred_props_file=None,
                                 word_dict=data.word_dict)
        batched_dev_data = data.get_development_data(batch_size=config.dev_batch_size)
        print ('Dev data has {} batches.'.format(len(batched_dev_data)))

    with Timer('Syntactic Information Extracting'):  # extract the syntactic information from file
        # Data for dep Trees
        train_dep_paths = args.train_dep_trees.split(';')
        dev_dep_paths = args.dev_dep_trees.split(';')
        dep_data_path_set = zip(train_dep_paths, dev_dep_paths)
        dep_treebanks_num = len(train_dep_paths)
        hete_deps = []
        for i in xrange(dep_treebanks_num):
            train_path, dev_path = dep_data_path_set[i]
            train_dep_trees, dev_dep_trees = SyntacticCONLL(), SyntacticCONLL()
            train_dep_trees.read_from_file(train_path)
            dev_dep_trees.read_from_file(dev_path)
            # generate the syntactic label dict in training corpus
            train_dep_trees.get_syntactic_label_dict(data.dep_label_dicts[i])
            dev_dep_trees.get_syntactic_label_dict(data.dep_label_dicts[i])
            ## append
            hete_deps.append((train_dep_trees, dev_dep_trees))

    with Timer('Preparation'):
        if not os.path.isdir(args.model):
            print ('Directory {} does not exist. Creating new.'.format(args.model))
            os.makedirs(args.model)
        else:
            if len(os.listdir(args.model)) > 0:
                print ('[WARNING] Log directory {} is not empty, previous checkpoints might be overwritten'
                       .format(args.model))
        shutil.copyfile(args.config, os.path.join(args.model, 'config'))
        # Save word and label dict to model directory.
        data.word_dict.save(os.path.join(args.model, 'word_dict'))
        data.head_dict.save(os.path.join(args.model, 'head_dict'))
        data.char_dict.save(os.path.join(args.model, 'char_dict'))
        data.label_dict.save(os.path.join(args.model, 'label_dict'))
        for i in xrange(len(data.dep_label_dicts)):
            data.dep_label_dicts[i].save(os.path.join(args.model, 'dep_label_dict' + str(i)))
        writer = open(os.path.join(args.model, 'checkpoints.tsv'), 'w')
        writer.write('step\tdatetime\tdev_loss\tdev_accuracy\tbest_dev_accuracy\n')

    with Timer('Building NN model'):
        model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
        if args.gpu:
            print "BiLSTMTaggerModel initialize with GPU!"
            model = model.to(device)
            if args.gpu != "" and not torch.cuda.is_available():
                raise Exception("No GPU Found!")
                exit()
        for name, param in model.named_parameters():  # print pytorch model parameters and the corresponding names
            print name, param.size()

    i, global_step, epoch, train_loss = 0, 0, 0, 0.0
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    last_lr = 0.001
    no_more_better_performance = 0
    optimizer = torch.optim.Adam(parameters, lr=last_lr)  # initialize the model parameter optimizer
    max_steps = int(config.max_steps)
    while global_step <= max_steps:  # epoch < config.max_epochs
        initial_time = time.time()
        with Timer("Epoch%d" % epoch) as timer:
            model.train()
            dep_train_data = data.get_dep_training_data(include_last_batch=True)
            train_data = data.get_training_data(include_last_batch=True)
            mixed_data = data.mix_training_data(train_data, dep_train_data)
            for batched_tensor, batched_dep_tensor in mixed_data:  # for each batch in the training corpus
                sent_ids, sent_lengths, \
                word_indexes, head_indexes, char_indexes, \
                predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens,\
                gold_predicates, num_gold_predicates = batched_tensor
                hete_dep_trees = get_hete_dep_trees_info(hete_deps, sent_ids, sent_lengths)

                if args.gpu:
                    word_indexes, head_indexes, char_indexes,\
                        predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens = \
                        word_indexes.cuda(), head_indexes.cuda(), char_indexes.cuda(), predicate_indexes.cuda(), arg_starts.cuda(), \
                        arg_ends.cuda(), arg_labels.cuda(), srl_lens.cuda()  # gold_predicates.cuda(), num_gold_predicates.cuda()

                optimizer.zero_grad()
                predicated_dict, srl_loss = model.forward(sent_lengths, word_indexes, head_indexes, char_indexes,
                                                      (predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens),
                                                      (gold_predicates, num_gold_predicates),
                                                      tree_gru_input=hete_dep_trees)
                srl_loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # dep forward
                dep_losses = []
                for ith, a_batched_dep_tensor in enumerate(batched_dep_tensor):
                    word_indexes, char_indexes, mask, lengths, heads, labels = a_batched_dep_tensor
                    if args.gpu:
                        word_indexes, char_indexes = word_indexes.cuda(), char_indexes.cuda()
                    dep_loss = model.forward(lengths, word_indexes, None, char_indexes,
                                             None, None, None,
                                             (ith, heads, labels))
                    dep_losses.append(dep_loss.detach())
                    optimizer.zero_grad()
                    dep_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                loss = srl_loss.detach() + sum(dep_losses)
                train_loss += float(loss.detach())  # should be tensor not Variable, avoiding the graph accumulates

                i += 1
                global_step += 1
                if global_step % 100 == 0:
                    last_lr = adjust_learning_rate(optimizer, last_lr)
                if i % 250 == 0:
                    total_time = time.time() - initial_time
                    timer.tick("{} training steps, loss={:.3f}, steps/s={:.2f}".format(
                        global_step, float(train_loss / i), float(global_step / total_time)))
                    train_loss = 0.0
                    i = 0

            train_loss = train_loss / i
            print("Epoch {}, steps={}, loss={:.3f}".format(epoch, i, float(train_loss)))

            i = 0
            epoch += 1
            train_loss = 0.0
            if epoch % config.checkpoint_every_x_epochs == 0:
                with Timer('Evaluation'):
                    evaluate_tagger(model, batched_dev_data, hete_deps, data.eval_data, data.label_dict, config, evaluator,
                                    writer, global_step)
                if evaluator.has_best is True:
                    no_more_better_performance = 0
                else:
                    no_more_better_performance += 1
                    if no_more_better_performance >= 200:
                        print("no more better performance since the past 200 epochs!")
                        exit()

    # Done. :)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        type=str,
                        default='',
                        required=True,
                        help='Config file for the neural architecture and hyper-parameters.')

    parser.add_argument('--span',
                        type=str,
                        default='span',
                        required=True,
                        help='Whether current experiments is for span-based SRL. Default True (span-based SRL).')

    parser.add_argument('--model',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the directory for saving model and checkpoints.')

    parser.add_argument('--train',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the training data, which is a single file in sequential tagging format.')

    parser.add_argument('--dep_trees',
                        type=str,
                        default="",
                        required=True,
                        help="Path to the dependency trees data, which is a single file in CoNLL format."
                        )

    parser.add_argument('--train_dep_trees',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the training auto dep trees, optional')

    parser.add_argument('--dev',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the devevelopment data, which is a single file in the sequential tagging format.')

    parser.add_argument('--dev_dep_trees',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the dev auto dep trees, optional')

    parser.add_argument('--task',
                        type=str,
                        help='Training task (srl or propid). Default is srl.',
                        default='SRL',
                        choices=['SRL', 'propid'])

    parser.add_argument('--gold',
                        type=str,
                        default='',
                        help='(Optional) Path to the file containing gold propositions (provided by CoNLL shared task).')

    parser.add_argument('--vocab',
                        type=str,
                        default='',
                        help='(Optional) A file containing the pre-defined vocabulary mapping. Each line contains a text string for the word mapped to the current line number.')

    parser.add_argument('--labels',
                        type=str,
                        default='',
                        help='(Optional) A file containing the pre-defined label mapping. Each line contains a text string for the label mapped to the current line number.')

    parser.add_argument('--gpu',
                        type=str,
                        default="",
                        help='(Optional) A argument that specifies the GPU id. Default use the cpu')
    parser.add_argument('--info',
                        type=str,
                        default="",
                        help='(Optional) A additional information that specify this program.')
    args = parser.parse_args()
    train_tagger(args)

