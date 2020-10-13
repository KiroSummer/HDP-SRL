from layer import *


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks
    return inputs.transpose(1, 0)


def _model_var(parameters, x):
    p = next(iter(filter(lambda p: p.requires_grad, parameters)))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)


# Biaffine scorer
class BiaffineScorer(nn.Module):
    def __init__(self, input_size, mlp_arc_size, mlp_rel_size, dep_label_space_size, config):
        super(BiaffineScorer, self).__init__()
        self.dep_label_space_size = dep_label_space_size
        self.config = config
        self.mlp_arc_dep = NonLinear(
            input_size=input_size,
            hidden_size=mlp_arc_size + mlp_rel_size,
            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size=input_size,
            hidden_size=mlp_arc_size + mlp_rel_size,
            activation=nn.LeakyReLU(0.1))

        self.total_num = int((mlp_arc_size + mlp_rel_size) / 100)
        self.arc_num = int(mlp_arc_size / 100)
        self.rel_num = int(mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(mlp_arc_size, mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(mlp_rel_size, mlp_rel_size, self.dep_label_space_size, bias=(True, True))

    def forward(self, lstm_out, dep, sent_lengths):
        if self.training:
            lstm_out = drop_sequence_sharedmask(lstm_out, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(lstm_out)
        x_all_head = self.mlp_arc_head(lstm_out)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)

        self.arc_logits, self.rel_logits = arc_logit, rel_logit_cond

        heads, rels = dep[0], dep[1]
        loss = self.compute_dep_loss(heads, rels, sent_lengths)  # compute the dep loss
        return loss

    def compute_dep_loss(self, true_arcs, true_rels, lengths):
        b, l1, l2 = self.arc_logits.size()
        index_true_arcs = _model_var(
            self.parameters(),
            pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64))
        true_arcs = _model_var(
            self.parameters(),
            pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64))

        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-1000] * (l2 - length))
            mask = _model_var(self.parameters(), mask)
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            masks.append(mask.transpose(0, 1))
        length_mask = torch.stack(masks, 0)
        arc_logits = self.arc_logits + length_mask

        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1, reduction="sum")

        size = self.rel_logits.size()
        output_logits = _model_var(self.parameters(), torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        b, l1, d = output_logits.size()
        true_rels = _model_var(self.parameters(), pad_sequence(true_rels, padding=-1, dtype=np.int64))

        rel_loss = F.cross_entropy(
            output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1, reduction="sum")

        loss = arc_loss + rel_loss
        return loss