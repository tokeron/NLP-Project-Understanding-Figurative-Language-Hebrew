import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer, BertConfig
from transformers import AutoModelForTokenClassification, BertTokenizerFast, TrainingArguments, EarlyStoppingCallback
import torch
from transformers.modeling_outputs import TokenClassifierOutput


class alephMelBert(nn.Module):
    def __init__(self, layers_for_cls=None, only_intermid_rep=False):
        super(alephMelBert, self).__init__()
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

        self.base_model.classifier = nn.Linear(in_features=768 * 2 * (base_layers_num+len(layers_for_cls)),
                                    out_features=3, bias=True).to(self.base_model.classifier.weight.device)

    def forward(self, labels, input_ids, token_type_ids, attention_mask, only_intermediate_representation):
        outputs = self.base_model.base_model(input_ids, attention_mask=attention_mask)
        embeddings_in_context = outputs.last_hidden_state  # shape = [batch size, number of words, word embedding]
        hidden_states = outputs.hidden_states
        embeddings_without_context = []
        # iterate over each index of the input_ids for the hole batch
        for input_id, curr_attention_mask in zip(input_ids.T, attention_mask.T):
            model_output = self.base_model.base_model(input_id.unsqueeze(0).T,
                                                      attention_mask=curr_attention_mask.unsqueeze(0).T)
            embeddings_without_context.append(model_output.last_hidden_state)

        # if self.layers_for_cls is not None:
        #     embeddings_in_context_intermid = []
        #     embeddings_without_context_intermid = []
        #     for i in self.layers_for_cls:
        #         embeddings_in_context_intermid.append(hidden_states[i])
        #         # TODO add support for more layers
        #         embeddings_without_context_intermid.append(...)
        loss_fct = nn.CrossEntropyLoss()
        full_logits = None
        for embedding_in_context, embedding_without_context in \
                zip(embeddings_in_context.transpose(0, 1), embeddings_without_context):
            embedding_without_context = embedding_without_context.squeeze(1)
            concatenated_embedding = torch.cat((embedding_in_context,
                                                embedding_without_context), dim=1)
            logits = self.base_model.classifier(concatenated_embedding)
            if full_logits is None:
                full_logits = logits
            else:
                full_logits = torch.cat((full_logits, logits), dim=0)
        loss = loss_fct(full_logits, labels.transpose(0, 1).reshape(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=full_logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
