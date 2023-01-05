import torch
from dataset.AmazonFashionDataset import AmazonFashionDataset 
from model.SentimentClassifier import SentimentClassifier
import yaml
from torch.utils.tensorboard import SummaryWriter

def my_collate(batch):
    data, target = zip(*batch)
    # data = torch.tensor(data)
    # print(data)
    # exit()
    data_lens = [len(x) for x in data]
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    print(data_padded)
    print(data_lens)
    #packed = torch.nn.utils.rnn.pack_padded_sequence(data_padded, data_lens, batch_first=True, enforce_sorted=False)
    return data_padded, target, data_lens
    print(packed)
    exit()
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    #data = torch.LongTensor(data)

    return batch #[data, target]

def main():
    yaml_name = "cfg/train_config.yaml"
    with open(yaml_name) as f:
        yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    writer = SummaryWriter()
    dataset = AmazonFashionDataset(cfg)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size=2, collate_fn = my_collate)

    model = SentimentClassifier(cfg["MODEL"]["EMBEDDING_SIZE"], cfg["MODEL"]["HIDDEN_SIZE"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    temp = 0
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        for i, (data, label, data_lens) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data, data_lens)
            #out = out.unsqueeze(0)
            print(out)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, temp )
            temp += 1





if __name__ == "__main__":
    main()