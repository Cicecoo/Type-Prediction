# -*- coding: utf-8 -*-

import math

import torch

from ncc.criterions import NccCriterion
from ncc.data.constants import EPS
from ncc.utils.logging import metrics


class TripletCriterion(NccCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.margin = self.task.args['optimization']['margin']

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample_size,
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, repr, labels_or_equal_ids):
        """
        Memory-efficient triplet loss.
        - If labels_or_equal_ids is a 1D tensor of class ids (length B), compute masks on the fly
          and avoid materializing a full BxB adjacency matrix.
        - If it's a 2D matrix (BxB), fall back to the original behavior.
        """
        if labels_or_equal_ids.dim() == 2:
            # Backward-compatible path (will consume O(B^2) memory)
            equal_ids = labels_or_equal_ids
            distance = torch.norm(repr.unsqueeze(dim=0) - repr.unsqueeze(dim=1), dim=-1, p=1)  # B x B
            max_pos_distance = (distance * equal_ids).max(dim=-1)[0]
            neg_filter = distance <= (max_pos_distance + self.margin).unsqueeze(dim=-1)
            pos_mask = equal_ids + torch.eye(*equal_ids.size()).type_as(distance)
            neg_filter = neg_filter * (1 - pos_mask)
            avg_neg_distance = (distance * neg_filter).sum(dim=-1) / (neg_filter.sum(dim=-1) + EPS)
            min_neg_distance = (distance + pos_mask * 99999).min(dim=-1)[0]
            pos_filter = (distance >= (min_neg_distance - self.margin).unsqueeze(dim=-1)).type_as(distance)
            pos_filter = pos_filter * equal_ids
            avg_pos_distance = (distance * pos_filter).sum(dim=-1) / (pos_filter.sum(dim=-1) + EPS)
            triplet_loss = 0.5 * torch.relu(avg_pos_distance - min_neg_distance + self.margin) + \
                           0.5 * torch.relu(max_pos_distance - avg_neg_distance + self.margin)
            triplet_loss = triplet_loss.sum()
            return triplet_loss, None

        # Memory-efficient path using label ids
        labels = labels_or_equal_ids
        B, D = repr.size(0), repr.size(1)
        device = repr.device

        # Choose a conservative chunk size to bound K*B*D memory
        # Aim for ~64MB per chunk: (K * B * D * 4 bytes) ~= 64e6 -> K ~= 64e6 / (B*D*4)
        # Clamp to at least 1 and at most B
        est = max(1, int(64_000_000 / max(1, B * D * 4)))
        chunk_size = min(B, max(1, est))

        total_loss = repr.new_zeros(())
        for start in range(0, B, chunk_size):
            end = min(B, start + chunk_size)
            anchor = repr[start:end]  # [K, D]
            # Pairwise L1 distance between anchors and all samples: [K, B]
            dist = torch.sum(torch.abs(anchor.unsqueeze(1) - repr.unsqueeze(0)), dim=-1)

            labels_chunk = labels[start:end].unsqueeze(1)  # [K, 1]
            eq = (labels_chunk == labels.unsqueeze(0))  # [K, B]

            # Exclude self-comparisons within the current chunk
            eye = torch.zeros((end - start, B), dtype=torch.bool, device=device)
            eye[:, start:end] = torch.eye(end - start, dtype=torch.bool, device=device)
            pos_mask = eq & (~eye)

            # Max positive distance per anchor
            # If no positives, use 0 as default to avoid -inf
            max_pos_distance = torch.where(
                pos_mask, dist, torch.full_like(dist, -1e9)
            ).max(dim=1).values

            # Negatives are those with different labels
            neg_mask = ~eq
            # Min negative distance per anchor
            min_neg_distance = torch.where(
                neg_mask, dist, torch.full_like(dist, 1e9)
            ).min(dim=1).values

            # Hard-negatives around the margin band
            neg_filter = neg_mask & (dist <= (max_pos_distance + self.margin).unsqueeze(1))
            neg_count = neg_filter.sum(dim=1).clamp_min(1)
            avg_neg_distance = torch.where(neg_filter, dist, torch.zeros_like(dist)).sum(dim=1) / neg_count

            # Hard-positives around the margin band
            pos_filter = pos_mask & (dist >= (min_neg_distance - self.margin).unsqueeze(1))
            pos_count = pos_filter.sum(dim=1).clamp_min(1)
            avg_pos_distance = torch.where(pos_filter, dist, torch.zeros_like(dist)).sum(dim=1) / pos_count

            loss_chunk = 0.5 * torch.relu(avg_pos_distance - min_neg_distance + self.margin) + \
                         0.5 * torch.relu(max_pos_distance - avg_neg_distance + self.margin)
            total_loss = total_loss + loss_chunk.sum()

        return total_loss, None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        # ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
