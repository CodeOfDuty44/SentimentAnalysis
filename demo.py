import torch
from model.SentimentClassifier import SentimentClassifier
import argparse
import yaml
import torchtext
from dataset.AmazonFashionDataset import AmazonFashionDataset 


def parse_args():
    parser = argparse.ArgumentParser(description='Test LightTrack')
    parser.add_argument('--resume', type=str, help='pretrained model')
    parser.add_argument('--sentence', type= str, default=None)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    yaml_name = "cfg/train_config.yaml"
    with open(yaml_name) as f:
        yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    model = SentimentClassifier(cfg["MODEL"]["EMBEDDING_SIZE"], cfg["MODEL"]["HIDDEN_SIZE"])
    model.load_state_dict(torch.load(args.resume))
    model = model.eval().cuda()
    tokenizer = torchtext.data.get_tokenizer("basic_english")
    glove = torchtext.vocab.GloVe(name='twitter.27B', dim=25)

    text = args.sentence
    text = tokenizer(text)
    text = glove.get_vecs_by_tokens(text).unsqueeze(0).cuda()
    out = model(text, torch.tensor([text.shape[1]]))
    print(out)
    label = torch.argmax(out)
    if label == 0:
        print("Sentiment is: positive")
    elif label == 1:
        print("Sentiment is: neutral")
    else:
        print("Sentiment is: negative")


def print_confusion_matrix():
    args = parse_args()
    yaml_name = "cfg/train_config.yaml"
    with open(yaml_name) as f:
        yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    model = SentimentClassifier(cfg["MODEL"]["EMBEDDING_SIZE"], cfg["MODEL"]["HIDDEN_SIZE"])
    model.load_state_dict(torch.load(args.resume))
    model = model.eval()
    model = model.cuda()
    test_dataset = AmazonFashionDataset(cfg, mode = "test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size=2, collate_fn = my_collate)
    conf_matrix = torch.zeros((3,3))
    for i, (data, label, data_lens) in enumerate(test_dataloader):
            out = model(data, data_lens) #instead of try check
            for j in range(out.shape[0]):
                predicted = torch.argmax(out[j])
                gt = torch.argmax(label[j])
                conf_matrix[predicted, gt] += 1
    conf_matrix = conf_matrix.type(torch. int64) 
    print(conf_matrix)


def my_collate(batch):
    data, target = zip(*batch)
    data_lens = [len(x) for x in data]
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    target = torch.stack(target,0)
    return data_padded, target, data_lens

if __name__ == '__main__':
    #print_confusion_matrix()
    main()
