import torch
import torchtext

class SentimentClassifier(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(SentimentClassifier, self).__init__()
        self.embedding = torchtext.vocab.GloVe(name='twitter.27B', dim=25)
        self.emdedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size = self.emdedding_size, hidden_size = self.hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size,64)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(64,3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, x_len):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed)
        lstm_out_padded, lstm_output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        slicer  = torch.tensor(x_len).unsqueeze(1) - 1
        slicer = slicer.expand(slicer.shape[0],64).unsqueeze(1).cuda()
        lstm_actual_out = torch.gather(lstm_out_padded, 1, slicer) # b,time, hidden -> b,1,hidden
        lstm_actual_out = lstm_actual_out.squeeze(1) # b,1,hidden -> b, hidden

        y1 = self.linear(lstm_actual_out)
        y2 = self.leaky_relu(y1)
        y3 = self.linear2(y2)
        y4 = self.softmax(y3)
        return y4