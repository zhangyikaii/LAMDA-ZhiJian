import torch
from torch import nn
from torch.nn import functional as F

from pytorch_metric_learning.miners import TripletMarginMiner

def merge(anchor_id, positive_id, negative_id):
    merged_anchor_id = []
    merged_positive_id = []
    last_a = last_p = -1
    for i in range(0, anchor_id.size()[0]):
        if anchor_id[i] != last_a or positive_id[i] != last_p:
            merged_anchor_id.append(anchor_id[i])
            merged_positive_id.append(positive_id[i])
            last_a, last_p = anchor_id[i], positive_id[i]

    assert(len(merged_anchor_id) == len(merged_positive_id))
    n_tuples = len(merged_anchor_id)
    merged_negative_id = []
    for i in range(0, n_tuples):
        x = anchor_id == merged_anchor_id[i]
        y = positive_id == merged_positive_id[i]
        merged_negative_id.append(negative_id[x & y])
    
    merged_anchor_id = torch.Tensor(merged_anchor_id).long()
    merged_positive_id = torch.Tensor(merged_positive_id).long()
    
    max_len = 0
    for i in range(0, n_tuples):
        max_len = max(max_len, merged_negative_id[i].size()[0])

    mask = []
    for i in range(0, n_tuples):
        raw_len = len(merged_negative_id[i])
        if raw_len < max_len:
            merged_negative_id[i] = torch.cat([merged_negative_id[i],
                torch.zeros(max_len - raw_len).long().cuda()])
        merged_negative_id[i] = merged_negative_id[i].unsqueeze(0)
        mask.append(torch.cat([
            torch.ones(raw_len).long().cuda(),
            torch.zeros(max_len - raw_len).long().cuda()
        ]).unsqueeze(0))
    merged_negative_id = torch.cat(merged_negative_id, dim=0)
    mask = torch.cat(mask, dim=0)

    return merged_anchor_id, merged_positive_id, merged_negative_id, mask

class Refilled_stage1(nn.Module):
    def __init__(self, temputer):
        super(Refilled_stage1, self).__init__()
        self.temputer = temputer
        self.miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')

    def forward(self, feat_s, feat_t, labels, flag_merge=False):

        loss_function = nn.KLDivLoss(reduction='batchmean')
        feat_s = F.normalize(feat_s, p=2, dim=1)
        feat_t = F.normalize(feat_t, p=2, dim=1)
        # use semi-hard triplet mining
        if not flag_merge:
            # generate triplets
            with torch.no_grad():
                anchor_id, positive_id, negative_id = self.miner(feat_s, labels)
            # get teacher embedding in triplets
            teacher_anchor = feat_t[anchor_id]
            teacher_positive = feat_t[positive_id]
            teacher_negative = feat_t[negative_id]
            # get student embedding in triplets
            student_anchor = feat_s[anchor_id]
            student_positive = feat_s[positive_id]
            student_negative = feat_s[negative_id]
            # get a-p dist and a-n dist in teacher embedding
            teacher_ap_dist = torch.norm(teacher_anchor - teacher_positive, p=2, dim=1)
            teacher_an_dist = torch.norm(teacher_anchor - teacher_negative, p=2, dim=1)
            # get a-p dist and a-n dist in student embedding
            student_ap_dist = torch.norm(student_anchor - student_positive, p=2, dim=1)
            student_an_dist = torch.norm(student_anchor - student_negative, p=2, dim=1)
            # get probability of triplets in teacher embedding
            teacher_prob = torch.sigmoid((teacher_an_dist - teacher_ap_dist) / self.temputer)
            teacher_prob_aug = torch.cat([teacher_prob.unsqueeze(1), 1 - teacher_prob.unsqueeze(1)])
            # get probability of triplets in student embedding
            student_prob = torch.sigmoid((student_an_dist - student_ap_dist) / self.temputer)
            student_prob_aug = torch.cat([student_prob.unsqueeze(1), 1 - student_prob.unsqueeze(1)])
            # compute loss function
            loss_value = 1000 * loss_function(torch.log(student_prob_aug), teacher_prob_aug)
        # use semi-hard tuple mining
        else:
            # renew loss function
            loss_function = nn.KLDivLoss(reduction='none')
            # generate tuples
            with torch.no_grad():
                anchor_id, positive_id, negative_id = self.miner(feat_s, labels)
                if flag_merge:
                    merged_anchor_id, merged_positive_id, merged_negative_id, mask = \
                        merge(anchor_id, positive_id, negative_id)
            # get teacher embedding in tuples
            teacher_anchor = feat_t[merged_anchor_id]
            teacher_positive = feat_t[merged_positive_id]
            teacher_negative = feat_t[merged_negative_id]
            # get student embedding in tuples
            student_anchor = feat_s[merged_anchor_id]
            student_positive = feat_s[merged_positive_id]
            student_negative = feat_s[merged_negative_id]
            # get a-p dist and a-n dist in teacher embedding
            teacher_ap_dist = torch.norm(teacher_anchor - teacher_positive, p=2, dim=1)
            teacher_an_dist = torch.norm(teacher_anchor.unsqueeze(1) - teacher_negative, p=2, dim=2)
            teacher_an_dist = torch.masked_fill(teacher_an_dist, mask == 0, 1e9)
            # get a-p dist and a-n dist in student embedding
            student_ap_dist = torch.norm(student_anchor - student_positive, p=2, dim=1)
            student_an_dist = torch.norm(student_anchor.unsqueeze(1) - student_negative, p=2, dim=2)
            student_an_dist = torch.masked_fill(student_an_dist, mask == 0, 1e9)
            # get logit of tuples in teacher embedding
            teacher_tuple_logit = torch.cat([-teacher_ap_dist.unsqueeze(1), -teacher_an_dist], dim=1) / self.temputer1
            # get logit of tuples in student embedding
            student_tuple_logit = torch.cat([-student_ap_dist.unsqueeze(1), -student_an_dist], dim=1) / self.temputer1

            # compute loss function
            loss_value = 1000 * loss_function(F.log_softmax(student_tuple_logit), F.softmax(teacher_tuple_logit))
            loss_value = torch.mean(torch.sum(loss_value, dim=1))

        return loss_value

class Refilled_stage2(nn.Module):
    def __init__(self, temputer2):
        super(Refilled_stage2, self).__init__()
        self.temputer2 = temputer2
        self.loss_func = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, out_s, out_t, n_classes, labels):
        label_table = torch.arange(n_classes).long().unsqueeze(1).cuda()
        class_in_batch = (labels == label_table).any(dim=1)

        loss = self.loss_func(
            F.log_softmax(out_s[:, class_in_batch] / self.temputer2),
            F.softmax(out_t[:, class_in_batch] / self.temputer2, dim=1)
        )
        return loss
        