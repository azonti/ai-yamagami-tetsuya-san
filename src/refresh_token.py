import json
import tweepy

if __name__ == "__main__":
    with open("../credentials.json", encoding="utf-8") as f:
        credentials = json.loads(f.read())
    with open("../tokens.json", encoding="utf-8") as f:
        tokens = json.loads(f.read())

    oauth2_user_handler = tweepy.OAuth2UserHandler(
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        redirect_uri="https://localhost",
        scope=["tweet.write", "offline.access"],
    )
    tokens = oauth2_user_handler.refresh_token(
        "https://api.twitter.com/2/oauth2/token",
        client_id=credentials["client_id"],
        refresh_token=tokens["refresh_token"],
        auth=oauth2_user_handler.auth,
    )

    with open("../tokens.json", 'w', encoding="utf-8") as f:
        f.write(json.dumps(tokens))
