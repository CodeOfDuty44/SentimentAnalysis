import torch
from dataset.AmazonFashionDataset import AmazonFashionDataset 
from model.SentimentClassifier import SentimentClassifier
import yaml
from torch.utils.tensorboard import SummaryWriter

def my_collate(batch):
    data, target = zip(*batch)
    data_lens = [len(x) for x in data]
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    target = torch.stack(target,0)
    return data_padded, target, data_lens

def main():
    yaml_name = "cfg/train_config.yaml"
    with open(yaml_name) as f:
        yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    writer = SummaryWriter()
    dataset = AmazonFashionDataset(cfg)
    val_dataset = AmazonFashionDataset(cfg, mode = "val")
    save_path = cfg["WEIGHT_PATH"]
    train_loader = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size=cfg["TRAIN"]["BATCH"], collate_fn = my_collate)

    model = SentimentClassifier(cfg["MODEL"]["EMBEDDING_SIZE"], cfg["MODEL"]["HIDDEN_SIZE"])
    #model.load_state_dict(torch.load(save_path + "/epoch_49.pth"))
    criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.05, 0.55, 0.4])) #weights are inversely proportional to the class distributions[0.1, 0.55, 0.35]
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["TRAIN"]["LR"], momentum=cfg["TRAIN"]["MOMENTUM"])
    temp = 0
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        print("Epoch: ", epoch)
        for i, (data, label, data_lens) in enumerate(train_loader):
            if i % 100 == 0:
                print("batch: ", i)
            optimizer.zero_grad()
            out = model(data, data_lens) #instead of try check
            loss = criterion(out, label)
            loss.backward()
            # if i % 11 == 0:
            #     print("out: ", out)
            #     print("label: ",  label)
            #     exit()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, temp )
            temp += 1

        accuracy = validate(cfg, model, val_dataset)
        print("Epoch: ", epoch)
        print("Accuracy: ", accuracy)

        save_path = "weights/epoch_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), save_path)


def validate(cfg, model, val_dataset):
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle = False, batch_size=cfg["VAL"]["BATCH"], collate_fn = my_collate)
    accuracy = 0
    for i, (data, label, data_lens) in enumerate(val_dataloader):
        out = model(data, data_lens)
        out = torch.argmax(out, 1)
        label = torch.argmax(label, 1)
        accuracy += torch.sum(torch.eq(out,label))
    return accuracy / val_dataset.__len__()

if __name__ == "__main__":
    main()