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
    # print(data_padded)
    # print(data_lens)
    target = torch.stack(target,0)
    # print(target)
    # print(type(target))
    # exit()
    return data_padded, target, data_lens

def main():
    yaml_name = "cfg/train_config.yaml"
    with open(yaml_name) as f:
        yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    writer = SummaryWriter()
    dataset = AmazonFashionDataset(cfg)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size=cfg["TRAIN"]["BATCH"], collate_fn = my_collate)

    model = SentimentClassifier(cfg["MODEL"]["EMBEDDING_SIZE"], cfg["MODEL"]["HIDDEN_SIZE"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["TRAIN"]["LR"], momentum=cfg["TRAIN"]["MOMENTUM"])
    temp = 0
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        for i, (data, label, data_lens) in enumerate(train_loader):
            while(True):
                optimizer.zero_grad()
                try:
                    out = model(data, data_lens)
                except:
                    print("data: ", data)
                    print("data.shape: ", data.shape)
                    print("i: ", i)
                #out = out.unsqueeze(0)
                loss = criterion(out, label)
                loss.backward()
                if i % 100 == 0:
                    print("out: ", out)
                    print("label: ",  label)
                # print(model.lstm.weight_ih_l0.grad)
                # print("*************")
                # print(model.linear.weight.grad)
                # exit()
                optimizer.step()
                writer.add_scalar("Loss/train", loss, temp )
                temp += 1





if __name__ == "__main__":
    main()