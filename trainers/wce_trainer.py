import wandb
from helpers.metric import compute_metrics_baseline, compute_metrics_melbert
from helpers.utils import *
from config.config_parser import *
from torch import nn
from transformers import Trainer


class EvalPrediction(object):
    def __init__(self, logits, labels):
        self.predictions = logits
        self.label_ids = labels


class WeightedLossTrainer(Trainer):
    def __init__(self, per_example_label=False, melbert=False,
                 use_more_layers=False, only_intermediate_representation=False, layers_for_classification=None,
                 non_metaphor_weight=1.0, metaphor_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers_for_classification = layers_for_classification
        self.per_example_label = per_example_label
        self.args.remove_unused_columns = False
        self.non_metaphor_weight = non_metaphor_weight
        self.metaphor_weight = metaphor_weight
        self.melbert = melbert
        self.only_intermediate_representation = only_intermediate_representation
        self.use_more_layers = use_more_layers

    def compute_loss(self, model, inputs, return_outputs=False):
        layers_for_classification = self.layers_for_classification
        non_metaphor_weight = self.non_metaphor_weight
        metaphor_weight = self.metaphor_weight
        only_intermediate_representation = self.only_intermediate_representation
        if self.per_example_label or (self.use_more_layers and len(layers_for_classification) > 0):
            model_output = model(only_intermediate_representation=only_intermediate_representation,
                                 **inputs)
        else:
            model_output = model(**inputs)
        logits = model_output.get('logits')
        labels = inputs.get("labels")
        # create a tensor and move to cuda
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([non_metaphor_weight, metaphor_weight, metaphor_weight]).to(logits.device))
        num_labels = 3
        if self.melbert:
            loss = loss_fct(logits, labels.transpose(0, 1).reshape(-1))
        elif self.per_example_label:
            loss = loss_fct(logits.view(-1, num_labels), labels[:, 0])
        else:
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        # add a dummy to the first dim of logits to be compatible with the original code in transformers
        if not self.melbert:
            logits = torch.cat((torch.zeros(1, logits.shape[1], logits.shape[2]).to(logits.device), logits), dim=0)
        return (loss, logits) if return_outputs else loss

