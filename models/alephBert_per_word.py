import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer, BertConfig
from transformers import AutoModelForTokenClassification, BertTokenizerFast, TrainingArguments, EarlyStoppingCallback
import torch
from transformers.modeling_outputs import TokenClassifierOutput


class AlephBertPerWord(nn.Module):
    def __init__(self, layers_for_cls=None, only_intermid_rep=False):
        super(AlephBertPerWord, self).__init__()
        self.layers_for_cls = layers_for_cls
        label_names = ["O", "B-metaphor", "I-metaphor"]
        id2label = {str(i): label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        config = BertConfig.from_pretrained("onlplab/alephbert-base",
                                            output_hidden_states=True,
                                            id2label=id2label,
                                            label2id=label2id)
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            'onlplab/alephbert-base',
            config=config,
        )
        base_layers_num = 1
        if only_intermid_rep:
            base_layers_num = 0

        self.base_model.classifier = nn.Linear(in_features=768 * (base_layers_num+len(layers_for_cls)),
                                    out_features=3, bias=True).to(self.base_model.classifier.weight.device)

    def forward(self, labels, input_ids, token_type_ids, attention_mask, only_intermediate_representation):
                #  word_input_ids, word_token_type_ids, word_attention_mask, word_idx):
        outputs = self.base_model.base_model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.get("last_hidden_state")
        hidden_states = outputs.get("hidden_states")
        classification_input = last_hidden_states
        # concatenate the hidden states of the encoder with the hidden states of the decoder
        if only_intermediate_representation:
            classification_input = hidden_states[self.layers_for_cls[0]]
        else:
            for layer in self.layers_for_cls:
                classification_input = torch.cat((classification_input, hidden_states[layer]), dim=-1)
        logits = self.base_model.classifier(classification_input[:, 0, :].unsqueeze(1))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels[:, 0])

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
