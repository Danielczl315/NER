import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class data_loader():
    def __init__(self, mode='train', padding=40):
        if mode == 'train':
            self.path = 'data/train/train.txt'
            self.isTest = False
        elif mode == 'dev':
            self.path = 'data/dev/dev.txt'
            self.isTest = False
        else:
            self.path = 'data/test/test.nolabels.txt'
            self.isTest = True
        self.padding = padding
        self.sentences = []
        self.vocab = set()
        self.labels = []
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        self.embeds = []
        self.embeds_labels = []
    
    def load(self):
        with open(self.path) as f:
            data = f.readlines()
            sentences = []
            labels = []
            vocab = set()
            new_sent = []
            new_label = []
            for token in data:
                if token == '\n':
                    self.sentences.append(new_sent)
                    self.labels.append(new_label)
                    new_sent = []
                    new_label = []
                else:
                    if self.isTest:
                        word = token.replace('\n','').split('\t')[0]
                        label = None
                    else:
                        word, label = token.replace('\n','').split('\t')
                    new_sent.append(word)
                    new_label.append(label)
                    self.vocab.add(word)
                    
    def toELMo(self):
        truncated_sentences = []
        truncated_labels = []
        if not self.isTest:
            for label, sentence in zip(self.labels, self.sentences):
                if len(sentence) > self.padding:
                    truncated_sentences.append(sentence[:self.padding])
                    truncated_labels.append(label[:self.padding])
                else:
                    truncated_sentences.append(sentence)
                    while len(label) < self.padding:
                        label.append("O")
                    truncated_labels.append(label)
            character_ids = batch_to_ids(truncated_sentences)
            self.embeds = self.elmo(character_ids)['elmo_representations'][0]
            self.embeds_label = truncated_labels
        else:
            for sentence in self.sentences:
                if len(sentence) > self.padding:
                    truncated_sentences.append(sentence[:self.padding])
                else:
                    truncated_sentences.append(sentence)
            character_ids = batch_to_ids(truncated_sentences)
            self.embeds = self.elmo(character_ids)['elmo_representations'][0]

        
        
class ELMO_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim, padding=40, embedding_dim=1024):
        super(ELMO_BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.padding = padding
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        # dropout layer
        self.dropout = nn.Dropout()
        
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.padding, self.hidden_dim // 2),
                torch.randn(2, self.padding, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        sentence = self.dropout(sentence)
        lstm_out, _ = self.lstm(sentence, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.padding, self.hidden_dim)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return (forward_score - gold_score).mean()

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    

    
    
class NERLearner():
    def __init__(self, model):
        self.model = model
        self.train_loss = []
       
    def fit(self, sentences, label, n_epochs=15):
        tag_to_ix = model.tag_to_ix
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in tqdm(range(n_epochs)): 
            for i in range(0, len(sentences)):
                model.zero_grad()

                sentence_input = sentences[i:i+1] # 1*40*1024
                targets = [tag_to_ix[t] for t in train_labels[i]] # 1*40
                targets = torch.tensor(targets, dtype=torch.long)

                loss = model.neg_log_likelihood(sentence_input, targets)
                loss.backward()
                optimizer.step()
            train_loss.append(loss.item())

    def predict(self, sentences):
        out = []
        for i in range(len(sentences)):
            sentence_input = sentences[i:i+1]
            with torch.no_grad():
                out.append(model(sentence))
        return out

import pickle
if __name__ == '__main__':
    train_loader = data_loader('train')
    train_loader.load()
    train_loader.toELMo()
    
    dev_loader = data_loader('dev')
    dev_loader.load()
    dev_loader.toELMo()
    
    test_loader = data_loader('test')
    test_loader.load()
    test_loader.toELMo()
                              
    HIDDEN_DIM = 680 # 2/3 * embedding dimension
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    model = ELMO_BiLSTM_CRF(tag_to_ix, HIDDEN_DIM)
    learner = NERLearner(model)
                           
    learner.fit(train_loader.embeds, train_loader.embeds_labels)                        
    pickle.dump(learner.model, open('model.pkl'))                    
                              
    