from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup
from metric import Metric
from models import (SpellBert, SpellBertPho1, SpellBertPho2, 
                        SpellBertPho1Res, SpellBertPho2Res, 
                        SpellBertPho2ResArch2, SpellBertPho2ResArch3,
                        SpellBertPho2ResArch4,)

import pickle, time

logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, SpellBert, BertTokenizer),
    'bert-pho1': (BertConfig, SpellBertPho1, BertTokenizer),
    'bert-pho2': (BertConfig, SpellBertPho2, BertTokenizer),
    'bert-pho1-res': (BertConfig, SpellBertPho1Res, BertTokenizer),
    'bert-pho2-res': (BertConfig, SpellBertPho2Res, BertTokenizer),
    'bert-pho2-res-arch2': (BertConfig, SpellBertPho2ResArch2, BertTokenizer),    
    'bert-pho2-res-arch3': (BertConfig, SpellBertPho2ResArch3, BertTokenizer),    
    'bert-pho2-res-arch4': (BertConfig, SpellBertPho2ResArch4, BertTokenizer),    
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dataset(args, input_file):
    input_file = os.path.join(args.data_dir, input_file)
    dataset = pickle.load(open(input_file, 'rb'))
    return dataset

import threading
from queue import Queue
class DataProcessor(threading.Thread):
    """docstring for DataProcessor"""
    def __init__(self, args, dataset, tokenizer, batch_processor, is_eval=False, max_queue_size=3):
        super(DataProcessor, self).__init__()
        self.args = args
        self.stream = torch.cuda.Stream()
        self.queue = Queue(maxsize=max_queue_size)
        self.device = args.device
        self.to_gpu_keys = args.to_gpu_keys
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_processor = batch_processor
        self.is_eval = is_eval

    def make_features(self, examples):
        max_length = self.args.max_seq_length
        batch = {}
        for t in ['id', 'src', 'tgt', 'tokens_size', 'lengths', 'src_idx', 'tgt_idx', 'masks', 'loss_masks']:
            batch[t] = []
        for item in examples:
            for t in item:
                if t == 'src_idx' or t == 'tgt_idx':
                    seq = item[t][:max_length]
                    padding_length = max_length - len(seq)
                    batch[t].append(seq + ([0]*padding_length))
                    if t == 'src_idx':
                        batch['masks'].append(([1]*len(seq)) + ([0]*padding_length))
                elif t == 'lengths':
                    batch[t].append(item[t])
                    loss_mask = [0] * max_length
                    for i in range(1, min(1+item[t], max_length)):
                        loss_mask[i] = 1
                    batch['loss_masks'].append(loss_mask)
                else:
                    batch[t].append(item[t])
        batch['src_idx'] = torch.tensor(batch['src_idx'], dtype=torch.long)
        batch['tgt_idx'] = torch.tensor(batch['tgt_idx'], dtype=torch.long)
        batch['masks'] = torch.tensor(batch['masks'], dtype=torch.long)
        batch['loss_masks'] = torch.tensor(batch['loss_masks'], dtype=torch.long)

        batch = self.batch_processor(batch, self.tokenizer)
        for key in self.to_gpu_keys:
            if key in batch:
                batch[key] = batch[key].to(device=self.device)
        return batch

    def run(self):
        data_size = len(self.dataset)
        if not self.is_eval:
            random.shuffle(self.dataset)
            bs = self.args.train_batch_size
        else:
            bs = self.args.eval_batch_size

        with torch.cuda.stream(self.stream):
            for i in range(0, data_size, bs):
                self.queue.put(self.make_features(self.dataset[i:min(i+bs,data_size)]))

def train(args, model, tokenizer, batch_processor):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    if args.local_rank == -1:
        train_dataset = create_dataset(args, args.train_file)
    else:
        total_dataset = create_dataset(args, args.train_file)
        start_position = 0
        width = torch.distributed.get_world_size()
        train_dataset = []
        while start_position + width <= len(total_dataset):
            train_dataset.append(total_dataset[start_position+args.local_rank])
            start_position += width 

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    need_optimized_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in need_optimized_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in need_optimized_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
            amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset) * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        data_helper = DataProcessor(args, train_dataset, tokenizer, batch_processor, False)
        data_helper.start()
        time.sleep(2)
        step = 0
        while not data_helper.queue.empty():
            batch = data_helper.queue.get()
            model.train()
            loss = model(batch)[0]
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
                    logger.info("Step: {}, LR: {}, Loss: {}".format(global_step, logs['learning_rate'], logs['loss']))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'saved_ckpt-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
            step += 1
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, batch_processor, prefix=""):
    eval_dataset = create_dataset(args, args.dev_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size 

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0

    batches = []
    data_helper = DataProcessor(args, train_dataset, tokenizer, batch_processor, False)
    data_helper.start()
    time.sleep(2)

    while not data_helper.queue.empty():
        batch = data_helper.queue.get()
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=-1)
        batch['src_idx'] = batch['src_idx'].detach().cpu().numpy()
        batch['pred_idx'] = preds
        batches.append(batch)

    metric = Metric(vocab_path=args.output_dir)
    pred_txt_path = os.path.join(args.output_dir, prefix, "preds.txt")
    pred_lbl_path = os.path.join(args.output_dir, prefix, "labels.txt")
    results = metric.metric(
            batches=batches,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=os.path.join(args.data_dir, args.dev_label_file)
        )
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--image_model_type", default=0, type=int)
    parser.add_argument("--model_name_or_path", default='/data/dobby_ceph_ir/neutrali/pretrained_models/roberta-base-ch-for-csc', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--font_path", default='/data/dobby_ceph_ir/neutrali/experiments/spell-acl/simhei.ttf', type=str)
    parser.add_argument("--data_dir", default="/data/dobby_ceph_ir/neutrali/experiments/spell-acl-data", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_file", default="train.pkl", type=str)
    parser.add_argument("--dev_file", default="dev.pkl", type=str)
    parser.add_argument("--dev_label_file", default="dev.lbl.tsv", type=str)
    parser.add_argument("--predict_file", default="test.sighan15.pkl", type=str)
    parser.add_argument("--predict_label_file", default="test.sighan15.lbl.tsv", type=str)
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--order_metric", default='avg_loss', type=str)
    parser.add_argument("--metric_reverse", action='store_true')
    parser.add_argument("--num_save_ckpts", default=5, type=int)
    parser.add_argument("--remove_unused_ckpts", action='store_true')
    
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.to_gpu_keys = ['src_idx', 'tgt_idx', 'masks', 'loss_masks', 'pho_idx', 'pho_lens']
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          image_model_type=args.image_model_type,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path, config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.tie_cls_weight()
    if args.model_type in ['bert-pho1-res', 'bert-pho2-res']:
        model.build_glyce_embed(args.model_name_or_path, args.font_path)
    batch_processor = model_class.build_batch

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, batch_processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        
        best_ckpt_dirs = []
        results = {}

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('saved_ckpt-') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, batch_processor, prefix=prefix)
            best_ckpt_dirs.append((result[args.order_metric], checkpoint))
            
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result) 

        best_ckpt_dirs = sorted(best_ckpt_dirs, reverse=args.metric_reverse)[:args.num_save_ckpts]
        best_ckpt_dirs = [d for v, d in best_ckpt_dirs]

        json.dump(results, open(os.path.join(args.output_dir, 'dev_results.json'), 'w', encoding='utf-8'), indent=4)

        if args.remove_unused_ckpts:
            for checkpoint in checkpoints:
                prefix = checkpoint.split('/')[-1] if checkpoint.find('saved_ckpt-') != -1 else ""
                if len(prefix) != 0 and (checkpoint not in best_ckpt_dirs):
                    logger.info("Deleting ckpt: %s", checkpoint)
                    os.system("rm -rf %s"%checkpoint)

    if args.do_predict and args.local_rank in [-1, 0]:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        
        args.dev_file = args.predict_file
        args.dev_label_file = args.predict_label_file
        results = {}

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('saved_ckpt-') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, batch_processor, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

        json.dump(results, open(os.path.join(args.output_dir, 'predict_results.json'), 'w', encoding='utf-8'), indent=4)


