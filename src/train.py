import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
    GPT2Config,
)

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from tweet_dataset import TweetDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "../pretrained",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    assert isinstance(tokenizer, PreTrainedTokenizer)
    model = GPT2LMHeadModel.from_pretrained(
        "../pretrained",
    )
    assert isinstance(model, GPT2LMHeadModel)
    config = model.config
    assert isinstance(config, GPT2Config)

    def transform(tweet: str) -> tuple[torch.Tensor, torch.Tensor]:
        tweet = tokenizer.bos_token+tweet

        model_input = tokenizer(
            tweet,
            padding="max_length",
            max_length=config.n_positions,
            return_tensors="pt"
        )

        input_ids = model_input["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        input_ids = input_ids[0]

        attention_mask = model_input["attention_mask"]
        assert isinstance(attention_mask, torch.Tensor)
        attention_mask = attention_mask[0]

        return input_ids, attention_mask

    train_dataset = TweetDataset(
        "../yamagami.html",
        train=True,
        transform=transform,
    )
    validate_dataset = TweetDataset(
        "../yamagami.html",
        train=False,
        transform=transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    pbar = tqdm(total=args.num_epochs*len(train_dataloader))
    for epoch in range(args.num_epochs):
        model.train()
        for batch, (input_ids, attention_mask) in enumerate(train_dataloader):
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            model_output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 99:
                print(f"Train Loss: {model_output.loss:>7f}")
            pbar.update(1)

        losses = torch.empty(0, dtype=torch.float64)
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask in validate_dataloader:
                model_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                losses = torch.cat((losses, model_output.loss), dim=0)
        print(f"Validate Loss: {losses.mean():>7f}")

        if not os.path.exists(f"../finetuned/epoch{epoch}"):
            os.makedirs(f"../finetuned/epoch{epoch}")

        tokenizer.save_pretrained(f"../finetuned/epoch{epoch}")
        tokenizer.save_vocabulary(f"../finetuned/epoch{epoch}")
        model.save_pretrained(f"../finetuned/epoch{epoch}")
