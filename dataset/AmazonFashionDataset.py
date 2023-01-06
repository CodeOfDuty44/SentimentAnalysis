import json
import yaml
import torch
import random
import torchtext

class AmazonFashionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, train=True):
        super(AmazonFashionDataset, self).__init__()
        self.dataPath = cfg["DATASET_PATH"]
        with open(self.dataPath) as f:
            temp = f.read()
            temp = temp.replace('\n', '')
            temp = temp.replace('}{', '},{')
            temp = "[" + temp + "]"
            self.data =  json.loads(temp)

        self.len = len(self.data)
        self.tokenizer = torchtext.data.get_tokenizer("basic_english")
        self.glove = torchtext.vocab.GloVe(name='twitter.27B', dim=25)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        try:
            #index=0
            text = self.data[index]["summary"]
            text = self.tokenizer(text)
            text = [self.glove.stoi[i] for i in text] #word2index
            text = torch.tensor(text)
            score = self.data[index]["overall"]
            label = torch.zeros(3)
            if score > 2:
                label[0] = 1
            elif score == 2:
                label[1] = 1
            else:
                label[2] = 1
            return text, label
        except:
            # print("Warning: Data could not be read, tried index is: ", index)
            i = random.randint(0, self.len - 1)
            # print("new index is: ", i)
            return self.__getitem__(i)



def main():
    # deneme = torchtext.vocab.GloVe(name='twitter.27B', dim=25)
    # print(deneme.stoi["back"])
    # print(deneme.vectors[deneme.stoi["back"]])
    # exit()
    yaml_name = "cfg/train_config.yaml"
    with open(yaml_name) as f:
        yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    dataset = AmazonFashionDataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle = False)
    datasetLen = dataset.__len__()
    corrupted = []
    reviewScores = [0] * 5

    glove = torchtext.vocab.GloVe(name='twitter.27B', dim=25)
    data, label = dataset.__getitem__(32169)
    print(data)
    print("aaa")
    ccc = dataset.data[21394]["summary"]
    ccc = dataset.tokenizer(ccc)
    print(ccc)
    exit()
    tokenizer = torchtext.data.get_tokenizer("basic_english")
    data = glove.get_vecs_by_tokens(data)
    print(data)
    exit()

    for i in range(datasetLen):
        try:
            data, label = dataset.__getitem__(i)
            print(type(data))
            reviewScores[int(label-1)] += 1
        except: 
            #print("Data with index i is corrupted: ", i)
            corrupted.append(i)
    print("corrputed entry size: ", len(corrupted))
    print("reviewScores: ", reviewScores)







if __name__ == "__main__":
    main()
