import argparse
import os
import sys

import torch
import torch.nn.functional as F

from ncc import LOGGER, tasks
from ncc.utils import checkpoint_utils, utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import progress_bar


results_path = './results/'


class SimpleTypePredictor:
    """Run each model on the graph batch and (optionally) average logits."""

    def __call__(self, models, sample):
        return self.predict(models, sample)

    def predict(self, models, sample):
        outputs = [model(**sample['net_input']) for model in models]
        if len(outputs) == 1:
            return outputs[0]

        first = outputs[0]
        if isinstance(first, torch.Tensor):
            stacked = torch.stack(outputs).mean(0)
            return stacked

        if isinstance(first, (tuple, list)) and isinstance(first[0], torch.Tensor):
            avg_logits = torch.stack([out[0] for out in outputs]).mean(0)
            rest = first[1:]
            return (avg_logits, *rest)

        return first


def main(args):
    if results_path is not None:
        os.makedirs(results_path, exist_ok=True)
        output_path = os.path.join(
            results_path, f"res.txt"
        )
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def accuracy(output, target, topk=(1,), ignore_idx=[]):
    with torch.no_grad():
        maxk = max(topk)

        # output: [num_nodes_with_annotations, vocab_size]
        target_vocab_size = output.size(-1)

        keep_idx = torch.tensor(
            [i for i in range(target_vocab_size) if i not in ignore_idx],
            device=output.device,
        ).long()
        _, pred = output[..., keep_idx].topk(maxk, -1, True, True)
        pred = keep_idx[pred]

        correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).long()
        mask = torch.ones_like(target).long()
        for idx in ignore_idx:
            mask = mask.long() & (~target.eq(idx)).long()
        mask = mask.long()
        deno = mask.sum().item()    # denominator: Acc@k 的分母
        correct = correct * mask.unsqueeze(-1)

        res = []
        for k in topk:
            res.append(correct[..., :k].reshape(-1).float().sum().item())
        return res, deno


def _main(args, _output_file):
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['test_subset'])

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        if _model_args['common']['fp16']:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args['dataset']['test_subset']),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        # max_positions=utils.resolve_max_positions(
        #     task.max_positions(),
        #     *[model.max_positions() for model in models]
        # ),
        # max_positions=utils.resolve_max_positions(
        #     getattr(task, 'max_positions', lambda: None)(),
        #     *[model.max_positions() for model in models],
        # ),
        ignore_invalid_inputs=_model_args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=_model_args['dataset']['required_batch_size_multiple'],
        num_shards=_model_args['dataset']['num_shards'],
        shard_id=_model_args['dataset']['shard_id'],
        num_workers=_model_args['dataset']['num_workers'],
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=_model_args['common']['log_format'],
        log_interval=_model_args['common']['log_interval'],
        default_log_format=('tqdm' if not _model_args['common']['no_progress_bar'] else 'none'),
    )

    # Typilus 推理
    type_predictor = SimpleTypePredictor()
    tgt_dict = task.target_dictionary(key='supernodes.annotation.type')
    no_type_idx = tgt_dict.indices.get('O', -100)
    any_type_idx = tgt_dict.indices.get('$any$', None)

    with torch.no_grad():
        # Accumulate metrics across batches to compute label-wise accuracy
        num1, num5, num_labels_total = 0, 0, 0
        num1_any, num5_any, num_labels_any_total = 0, 0, 0
        total_loss = 0
        count = 0

        for sample in progress:
            if use_cuda:
                sample = utils.move_to_cuda(sample)
            if 'net_input' not in sample:
                continue

            net_output = type_predictor.predict(models, sample)

            logits = net_output[0] if isinstance(net_output, (tuple, list)) else net_output
            # 只对有标注的节点进行评估
            if 'target_ids' in sample:
                logits = logits.index_select(0, sample['target_ids'])
            labels = sample['target'].view(-1)

            # Compute loss
            loss = F.cross_entropy(logits, labels, ignore_index=no_type_idx)
            total_loss += loss.item()

            # Accuracy helper expects a batch dimension, so wrap the flat tensors
            logits_metric = logits.unsqueeze(0)
            labels_metric = labels.unsqueeze(0)
            (corr1_any, corr5_any), num_labels_any = accuracy(
                logits_metric.cpu(), labels_metric.cpu(), topk=(1, 5), ignore_idx=(no_type_idx,)
            )
            num1_any += corr1_any
            num5_any += corr5_any
            num_labels_any_total += num_labels_any

            (corr1, corr5), num_labels = accuracy(
                logits_metric.cpu(), labels_metric.cpu(), topk=(1, 5),
                ignore_idx=tuple(idx for idx in (no_type_idx, any_type_idx) if idx is not None),
            )
            num1 += corr1
            num5 += corr5
            num_labels_total += num_labels
            count += 1
            if count % 100 == 0:
                LOGGER.info('count: {}\t'.format(count))

        # Average accuracies
        avg_loss = float(total_loss)/count
        acc1 = float(num1) / num_labels_total * 100
        acc5 = float(num5) / num_labels_total * 100
        acc1_any = float(num1_any) / num_labels_any_total * 100
        acc5_any = float(num5_any) / num_labels_any_total * 100

        LOGGER.info('avg_loss: {}\t acc1: {}\t acc5: {}\t acc1_any: {}\t acc5_any: {}'.format(avg_loss, acc1, acc5, acc1_any, acc5_any))

        # Write results to file
        _output_file.write('avg_loss: {}\n'.format(avg_loss))
        _output_file.write('acc1: {}\n'.format(acc1))
        _output_file.write('acc5: {}\n'.format(acc5))
        _output_file.write('acc1_any: {}\n'.format(acc1_any))
        _output_file.write('acc5_any: {}\n'.format(acc5_any))
        _output_file.close()

def cli_main():
    parser = argparse.ArgumentParser(description='Typilus type prediction')
    parser.add_argument(
        "--yaml_file", "-f", type=str, default='config/typilus',
        help="load {yaml_file}.yml relative to this directory",
    )
    parsed = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f'{parsed.yaml_file}.yml')
    LOGGER.info('Load arguments in %s', yaml_file)
    args = load_yaml(yaml_file)
    LOGGER.info(args)


    main(args)


if __name__ == '__main__':
    cli_main()
