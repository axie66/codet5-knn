import torch
from torch.utils import data
import argparse
from dataset import load_and_cache_gen_data, collate_fn
from config import add_args, parse_args
from transformers import RobertaTokenizer
from model import T5ForConditionalGeneration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parse_args(parser)

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

    _, train_data = load_and_cache_gen_data(args, args.test_filename, tokenizer, 'test')

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn)

    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    match = model.load_state_dict(torch.load('checkpoint/finetuned_models_concode_codet5_base.bin', map_location='cpu'))
    print(match)

    x, y, y_len = train_data[0]
    x = torch.tensor(x).unsqueeze(0)
    y = torch.tensor(y).unsqueeze(0)

    out = model(input_ids=x, labels=y, ret_decoder_ffn_inp=True)

    out = model.generate(
        input_ids=x,
        attention_mask=
    )
