import json
import tweepy

if __name__ == "__main__":
    with open("../credentials.json", encoding="utf-8") as f:
        credentials = json.loads(f.read())

    oauth2_user_handler = tweepy.OAuth2UserHandler(
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        redirect_uri="https://localhost",
        scope=[
            "users.read",
            "tweet.read",
            "tweet.write",
            "offline.access",
        ],
    )
    print(oauth2_user_handler.get_authorization_url())
    authorization_response_url = input("Authorization Response URL:")
    tokens = oauth2_user_handler.fetch_token(authorization_response_url)

    with open("../tokens.json", 'w', encoding="utf-8") as f:
        f.write(json.dumps(tokens))
