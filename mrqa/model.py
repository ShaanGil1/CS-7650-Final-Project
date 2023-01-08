import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertConfig
from transformers import DistilBertModel, DistilBertConfig
from utils import kl_coef


class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=3, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class DomainQA(nn.Module):
    def __init__(self, bert_name_or_config, num_classes=3, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False):
        super(DomainQA, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.config = self.bert.config

        self.qa_outputs = nn.Linear(hidden_size, 2)
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()
        if concat:
            input_size = 2 * hidden_size
        else:
            input_size = hidden_size
        self.discriminator = DomainDiscriminator(num_classes, input_size, hidden_size, num_layers, dropout)

        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        self.concat = concat
        self.sep_id = 102

    # only for prediction
    def forward(self, input_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None,
                dtype=None, global_step=22000):
        if dtype == "qa":
            qa_loss = self.forward_qa(input_ids, attention_mask,
                                      start_positions, end_positions, global_step)
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
            dis_loss = self.forward_discriminator(input_ids, attention_mask, labels)
            return dis_loss

        else:
            sequence_output = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = sequence_output.last_hidden_state
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            return start_logits, end_logits

    def forward_qa(self, input_ids, attention_mask, start_positions, end_positions, global_step):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = sequence_output.last_hidden_state
        print("Sequence output: ", sequence_output.shape)
        cls_embedding = sequence_output[:, 0]
        if self.concat:
            sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
            hidden = torch.cat([cls_embedding, sep_embedding], dim=1)
        else:
            hidden = sequence_output[:, 0]  # [b, d] : [CLS] representation
        log_prob = self.discriminator(hidden)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor. The targets are given as probabilities
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        if self.anneal:
            self.dis_lambda = self.dis_lambda * kl_coef(global_step)
        kld = self.dis_lambda * kl_criterion(log_prob, targets)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            sequence_output = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = sequence_output.last_hidden_state
            print("Sequence output: ", sequence_output.shape)
            cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
            print("CLS: ", cls_embedding.shape)
            if self.concat:
                sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
                hidden = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
            else:
                hidden = cls_embedding
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)

        return loss

    def get_sep_embedding(self, input_ids, sequence_output):
        batch_size = input_ids.size(0)
        sep_idx = (input_ids == self.sep_id).sum(1)
        sep_embedding = sequence_output[torch.arange(batch_size), sep_idx]
        return sep_embedding


class DomainQAMotivational(DomainQA):
    # New network with motivational discriminator
    def __init__(self, bert_name_or_config, num_classes=3, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False):
        super(DomainQAMotivational, self).__init__(bert_name_or_config, num_classes, hidden_size,
                 num_layers, dropout, dis_lambda, concat, anneal)

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.config = self.bert.config

        self.qa_outputs = nn.Linear(hidden_size, 2)
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()

        self.z_size = 512  # Domain invariant size
        self.a_size = 256  # Domain specific size
        self.discriminator = DomainDiscriminator(num_classes, self.z_size, hidden_size, num_layers, dropout)
        self.motivational = DomainDiscriminator(num_classes, self.a_size, hidden_size, num_layers, dropout)
        self.motivational_lambda = dis_lambda

        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        self.concat = concat
        self.sep_id = 102

        # Reshape z to concatenate with rest of sequence output
        self.fc = nn.Linear(self.z_size, hidden_size)

    # only for prediction
    def forward(self, input_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None,
                dtype=None, global_step=22000):
        if dtype == "qa":
            qa_loss = self.forward_qa(input_ids, attention_mask,
                                      start_positions, end_positions, global_step, labels)
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
            dis_loss = self.forward_discriminator(input_ids, attention_mask, labels)
            return dis_loss

        elif dtype == "motivational":
            assert labels is not None
            motivational_loss = self.forward_motivational(input_ids, attention_mask, labels)
            return motivational_loss

        else:
            sequence_output = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = sequence_output.last_hidden_state
            sequence_output = self.reshape_sequence(sequence_output)
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            return start_logits, end_logits

    def forward_qa(self, input_ids, attention_mask, start_positions, end_positions, global_step, labels):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = sequence_output.last_hidden_state
        cls_embedding = sequence_output[:, 0]
        z = cls_embedding[:, 256:].clone()  # Domain invariant
        a = cls_embedding[:, :256].clone()  # Domain specific
        log_prob = self.discriminator(z)
        log_prob_motivational = self.motivational(a)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor. The targets are given as probabilities
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        if self.anneal:
            self.dis_lambda = self.dis_lambda * kl_coef(global_step)
        kld = self.dis_lambda * kl_criterion(log_prob, targets)  # Domain invariant part

        # Reshape labels to expected one hot
        labels = F.one_hot(labels, self.num_classes).float()

        # Domain specific part
        kld_motivational = self.motivational_lambda * kl_criterion(log_prob_motivational, labels)

        sequence_output = self.reshape_sequence(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld + kld_motivational
        return total_loss

    def forward_discriminator(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            sequence_output = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = sequence_output.last_hidden_state
            cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
            z = cls_embedding[:, 256:].clone()  # Domain invariant
            hidden = z
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)

        return loss

    def forward_motivational(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            sequence_output = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = sequence_output.last_hidden_state
            cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
            a = cls_embedding[:, :256].clone()  # Domain specific
            hidden = a
        log_prob = self.motivational(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)

        return loss

    def reshape_sequence(self, sequence_output):
        cls_embedding = sequence_output[:, 0]
        z = cls_embedding[:, 256:].clone()  # Domain invariant
        z_reshaped = self.fc(z)
        sequence_output[:, 0] = z_reshaped  # Only use invariant part to compute output logits

        return sequence_output
