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
        #x_embed = self.embedding.get_vecs_by_tokens(x)
        
        x_embed = self.embedding.vectors[x]
        x_embed = torch.FloatTensor(x_embed)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_embed, x_len, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed)
        lstm_out_padded, lstm_output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        print(lstm_out)
        print(lstm_out_padded[0][4])
        slicer = [i-1 for i in x_len]
        slicer = torch.tensor([4, 52])
        print("slicer" , slicer)
        print("before" , lstm_out_padded.shape)
        lstm_actual_out = lstm_out_padded[:, slicer,:]
        lstm_actual_out = torch.index_select(lstm_out_padded, 1, slicer)
        print(lstm_actual_out.shape)
        exit()
        y1 = self.linear(lstm_out[:, -1])
        y2 = self.leaky_relu(y1)
        y3 = self.linear2(y2)
        y4 = self.softmax(y3)
        return y4

        # y1, dummy = self.lstm(x.float())
        # y1 = y1[:,149,:]
        # y2 = self.second(y1)
        # y3 = self.softmax(y2)
        # return y3
        # print("ccc")