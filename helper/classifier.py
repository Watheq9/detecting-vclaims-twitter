# -*- coding: utf-8 -*-
from torch import nn
from transformers import AutoModel


class RelevanceClassifierOneLayer(nn.Module):
    def __init__(self, bert_name, n_classes, freeze_bert=True, dropout=0.3, is_output_probability=True):

        super(RelevanceClassifierOneLayer, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_name)

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)

        self.pooler = BertPooler(self.bert.config)  # to be used in case of DistilBertModel

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(int(self.bert.config.hidden_size), n_classes)

        #softmax activation function
        self.softmax = nn.Softmax(dim=1)

        # whether to apply softmax on the output or not
        self.is_output_probability= is_output_probability

        #Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    #define the forward pass
    # def forward(self, inputs):
    def forward(self, input_ids, attention_mask):

        #pass the inputs to the model
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
        #  cls_output which is pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) 
        #  further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        bert_output = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              return_dict=False)
        
        if len(bert_output) == 1: # this case happens in DistilBertModel (some sentence transformers)
            last_hidden_state = bert_output[0]
            cls_output = self.pooler(last_hidden_state)
        else: # len should be two as it contains last hidden state and pooler output    
            last_hidden_state, cls_output = bert_output


        # output layer
        x = self.fc1(cls_output)
        x = self.dropout(x)

        if self.is_output_probability:
            # apply softmax activation
            x = self.softmax(x)

        return x


class RelevanceClassifierTwoLayers(nn.Module):
    def __init__(self, bert_name, n_classes, freeze_bert=True, dropout=0.3, is_output_probability=True):

        super(RelevanceClassifierTwoLayers, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_name)

        # self.pooler = BertPooler(self.bert.config)  # to be used in case of DistilBertModel

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        first_layer_neurons_count = int(self.bert.config.hidden_size)
        second_layer_neurons_count = int(first_layer_neurons_count/3)
        self.fc1 = nn.Linear(first_layer_neurons_count, second_layer_neurons_count)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(second_layer_neurons_count, n_classes)

        #softmax activation function
        self.softmax = nn.Softmax(dim=1)

        # whether to apply softmax on the output or not
        self.is_output_probability= is_output_probability

        #Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    #define the forward pass
    # def forward(self, inputs):
    def forward(self, input_ids, attention_mask):

        
        #pass the inputs to the model
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
        #  cls_output which is pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) 
        #  further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        bert_output = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              return_dict=False)
        
        if len(bert_output) == 1: # this case happens in DistilBertModel (some sentence transformers)
            last_hidden_state = bert_output[0]
            # pooler = BertPooler(self.bert.config)  # to be used in case of DistilBertModel
            cls_output = self.pooler(last_hidden_state)
        else: # len should be two as it contains last hidden state and pooler output    
            last_hidden_state, cls_output = bert_output

        x = self.fc1(cls_output)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        if self.is_output_probability:
            x = self.softmax(x)

        return x


class DuoBERTClassifierOneLayer(nn.Module):
    def __init__(self, bert_name, n_classes, freeze_bert=True, dropout=0.3,  is_output_probability=False):

        super(DuoBERTClassifierOneLayer, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_name)

        self.pooler = BertPooler(self.bert.config)  # to be used in case of DistilBertModel

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        first_layer_neurons_count = int(self.bert.config.hidden_size)
        self.fc1 = nn.Linear(first_layer_neurons_count, n_classes)

        #softmax activation function
        self.softmax = nn.Softmax(dim=1)

        # whether to apply softmax on the output or not
        self.is_output_probability= is_output_probability

        #Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    #define the forward pass
    # def forward(self, inputs):
    def forward(self, input_ids, attention_mask):

        #pass the inputs to the model
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
        #  cls_output which is pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) 
        #  further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        bert_output = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              return_dict=False)
        
        if len(bert_output) == 1: # this case happens in DistilBertModel (some sentence transformers)
            last_hidden_state = bert_output[0]
            cls_output = self.pooler(last_hidden_state)
        else: # len should be two as it contains last hidden state and pooler output    
            last_hidden_state, cls_output = bert_output


        # output layer
        x = self.fc1(cls_output)
        x = self.dropout(x)

        if self.is_output_probability:
            # apply softmax activation
            x = self.softmax(x)

        return x


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DuoBERTClassifierTwoLayers(nn.Module):
    def __init__(self, bert_name, n_classes, freeze_bert=True, dropout=0.3,  is_output_probability=False):

        super(DuoBERTClassifierTwoLayers, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_name)
        self.pooler = BertPooler(self.bert.config)  # to be used in case of DistilBertModel

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        first_layer_neurons_count = int(self.bert.config.hidden_size)
        second_layer_neurons_count = int(first_layer_neurons_count/3)
        self.fc1 = nn.Linear(first_layer_neurons_count, second_layer_neurons_count)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(second_layer_neurons_count, n_classes)

        #softmax activation function
        self.softmax = nn.Softmax(dim=1)

        # whether to apply softmax on the output or not
        self.is_output_probability= is_output_probability

        #Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    #define the forward pass
    def forward(self, input_ids, attention_mask):

        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
        #  cls_output which is pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) 
        #  further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        bert_output = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              return_dict=False)

        if len(bert_output) == 1: # this case happens in DistilBertModel (some sentence transformers)
            last_hidden_state = bert_output[0]
            pooler = BertPooler(self.bert.config)  # to be used in case of DistilBertModel
            cls_output = pooler(last_hidden_state)
        else: # len should be two as it contains last hidden state and pooler output    
            last_hidden_state, cls_output = bert_output

        x = self.fc1(cls_output)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        if self.is_output_probability:
            # apply softmax activation
            x = self.softmax(x)

        return x