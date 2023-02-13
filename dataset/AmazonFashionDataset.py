import json
import yaml
import torch
import random
import torchtext

class AmazonFashionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode="train"):
        super(AmazonFashionDataset, self).__init__()
        self.dataPath = cfg["DATASET_PATH"]
        with open(self.dataPath) as f:
            temp = f.read()
            temp = temp.replace('\n', '')
            temp = temp.replace('}{', '},{')
            temp = "[" + temp + "]"
            self.data =  json.loads(temp)
        if mode == "train":
            self.len = cfg["DATA"]["TRAIN_SIZE"]
            self.index_offset = 0
        elif mode == "val":
            self.len = cfg["DATA"]["VAL_SIZE"]
            self.index_offset = cfg["DATA"]["TRAIN_SIZE"]
        elif mode == "test":
            self.len = cfg["DATA"]["TEST_SIZE"]
            self.index_offset = cfg["DATA"]["TRAIN_SIZE"] + cfg["DATA"]["VAL_SIZE"] 
        self.tokenizer = torchtext.data.get_tokenizer("basic_english")
        self.glove = torchtext.vocab.GloVe(name='twitter.27B', dim=25)
        self.corrupted = 0

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        try:
            index += self.index_offset
            text = self.data[index]["summary"]
            text = self.tokenizer(text)
            text = self.glove.get_vecs_by_tokens(text)
            score = self.data[index]["overall"]
            label = torch.zeros(3)
            if score > 2:
                label[0] = 1
            elif score == 2:
                label[1] = 1
            else:
                label[2] = 1
            return text.cuda(), label.cuda()
        except Exception as e:
            self.corrupted += 1
            # print(e)
            # print("Warning: Data could not be read, tried index is: ", index)
            i = random.randint(0, self.len - 1)
            # print("new index that will be used is: ", i)
            return self.__getitem__(i)



def main():
    yaml_name = "cfg/train_config.yaml"
    with open(yaml_name) as f:
        yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    dataset = AmazonFashionDataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle = False)
    datasetLen = dataset.__len__()
    corrupted = []
    reviewScores = [0] * 3

    print("5 stars: ", dataset.__getitem__(1538))

    for i in range(datasetLen):
        try:
            data, label = dataset.__getitem__(i)
            label = torch.argmax(label)
            reviewScores[label] += 1
        except: 
            print("Data with index i is corrupted: ", i)
            corrupted.append(i)
    print("corrputed entry size: ", len(corrupted))
    print("reviewScores: ", reviewScores)








if __name__ == "__main__":
    main()
