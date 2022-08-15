from typing import TypeVar, Generic
from collections.abc import Callable

from torch.utils.data import Dataset
from bs4 import BeautifulSoup


T = TypeVar("T")


class TweetDataset(Dataset, Generic[T]):

    def __init__(
        self,
        html_path: str,
        train: bool,
        transform: Callable[[str], T] | None = None,
    ) -> None:

        self.html_path = html_path
        self.train = train
        self.transform = transform

        with open(html_path, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read())

        lis = soup.find_all("li")
        for li in lis:
            li.span.extract()

        tweets: list[str] = [li.get_text() for li in lis]
        tweets = [tweet for tweet in tweets if not tweet.startswith("RT")]
        tweets = [tweet.replace('\n    ', '\n') for tweet in tweets]
        tweets = [tweet.rstrip() for tweet in tweets]
        if train:
            tweets = [tweets[i] for i in range(0, len(tweets)) if i % 10 != 9]
        else:
            tweets = [tweets[i] for i in range(0, len(tweets)) if i % 10 == 9]

        self.tweets = tweets

    def __len__(self) -> int:
        return len(self.tweets)

    def __getitem__(self, idx: int) -> str | T:
        tweet = self.tweets[idx]
        if self.transform:
            tweet = self.transform(tweet)
        return tweet
