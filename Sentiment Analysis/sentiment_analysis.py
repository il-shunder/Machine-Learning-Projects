from newspaper import Article
from textblob import TextBlob

url = "https://en.wikipedia.org/wiki/AI_boom"

article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary
print(text)

blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)
