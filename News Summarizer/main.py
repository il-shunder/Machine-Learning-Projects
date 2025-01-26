import tkinter as tk

from newspaper import Article
from textblob import TextBlob


def summarize():
    url = url_text.get("1.0", "end").strip()

    if url:
        article = Article(url)

        article.download()
        article.parse()
        article.nlp()

        analysis = TextBlob(article.text)
        polarity = analysis.sentiment.polarity

        enable_text_fields()

        set_text_field(title, article.title)
        set_text_field(author, article.authors)
        set_text_field(date, article.publish_date)
        set_text_field(summary, article.summary)
        set_text_field(sentiment, get_sentiment_text(polarity))

        disable_text_fields()


def disable_text_fields():
    disable_element(title)
    disable_element(author)
    disable_element(date)
    disable_element(summary)
    disable_element(sentiment)


def disable_element(element):
    element.config(state="disabled")


def enable_text_fields():
    enable_element(title)
    enable_element(author)
    enable_element(date)
    enable_element(summary)
    enable_element(sentiment)


def enable_element(element):
    element.config(state="normal")


def set_text_field(field, text):
    if text:
        field.delete("1.0", "end")
        field.insert("1.0", text)


def get_sentiment_text(polarity):
    return (
        f"Polarity: {polarity}. Sentiment: {"positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"}"
    )


window = tk.Tk()
window.title("News Summarizer")
window.geometry("600x600")

title_label = tk.Label(window, text="Title")
title_label.pack()

title = tk.Text(window, height=1, width=160)
title.config(state="disabled", bg="#000", fg="#fff")
title.pack()

author_label = tk.Label(window, text="Author")
author_label.pack()

author = tk.Text(window, height=1, width=160)
author.config(state="disabled", bg="#000", fg="#fff")
author.pack()

date_label = tk.Label(window, text="Publishing Date")
date_label.pack()

date = tk.Text(window, height=1, width=160)
date.config(state="disabled", bg="#000", fg="#fff")
date.pack()

summary_label = tk.Label(window, text="Summary")
summary_label.pack()

summary = tk.Text(window, height=20, width=160)
summary.config(state="disabled", bg="#000", fg="#fff")
summary.pack()

sentiment_label = tk.Label(window, text="Sentiment Analysis")
sentiment_label.pack()

sentiment = tk.Text(window, height=1, width=160)
sentiment.config(state="disabled", bg="#000", fg="#fff")
sentiment.pack()

url_label = tk.Label(window, text="URL")
url_label.pack()

url_text = tk.Text(window, height=1, width=160)
url_text.config(bg="#fff", fg="#000")
url_text.pack()

button = tk.Button(window, text="Summarize", command=summarize)
button.pack()

window.mainloop()
