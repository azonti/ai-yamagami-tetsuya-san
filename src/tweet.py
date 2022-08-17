import argparse
import json
import tweepy
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
)


def generate(
    tokenizer: PreTrainedTokenizer,
    model: GPT2LMHeadModel,
    input_str: str,
) -> str:
    model_input = tokenizer(input_str, return_tensors="pt")
    model_output = model.generate(
        **model_input,
        max_new_tokens=280,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )
    output_str = tokenizer.decode(model_output[0].tolist())
    return output_str


def tweet(
    access_token: str,
    text: str,
) -> None:
    client = tweepy.Client(access_token)
    client.create_tweet(
        text=text,
        user_auth=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "../pretrained",
        trust_remote_code=True,
    )
    assert isinstance(tokenizer, PreTrainedTokenizer)
    model = GPT2LMHeadModel.from_pretrained(
        f"../finetuned/epoch{args.epoch}",
    )
    assert isinstance(model, GPT2LMHeadModel)

    while True:
        text = generate(tokenizer, model, tokenizer.bos_token)
        text = text.replace(tokenizer.bos_token, "")
        text = text.replace(tokenizer.eos_token, "")
        text = text.replace("<EMAIL>", "")
        text = text.replace("<URL>", "")
        text = text.strip()
        if text != "":
            break

    with open("../tokens.json", encoding="utf-8") as f:
        tokens = json.loads(f.read())

    tweet(tokens["access_token"], text)
