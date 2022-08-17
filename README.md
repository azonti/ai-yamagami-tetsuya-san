# ai-yamagami-tetsuya-san

## How to finetune your Japanese GPT-2 model

1. Copy your Japanese GPT-2 model to `pretrained` directory.
2. Run `curl https://vh8221.github.io/yamagami/ > yamagami.html`.
3. Run the below script.

```
pip install -r requirements.txt
pushd src
python -m train
popd
```

## How to generate a text & tweet

Firstly, make `credentials.json` like:

```
{
  "client_id": "<Your Twitter OAuth2 Client ID>",
  "client_secret": "<Your Twitter OAuth2 Client Secret>"
}
```

Secondly, do the authorization.

```
pip install -r requirements.txt
pushd src
python -m authorize
popd
```

Finally, generate a text & tweet!

```
pushd src
python -m tweet
popd
```
