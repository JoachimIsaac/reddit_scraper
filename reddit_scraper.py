import praw
import pandas as pd
import datetime
import os
import time
import spacy
import warnings
from openpyxl import load_workbook
from dotenv import load_dotenv
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

load_dotenv()

real_world_entity_parser = spacy.load("en_core_web_sm")

class RedditScraper:
    def __init__(self, subreddit, topics, max_posts=100, max_comments=50):
        self.subreddit_name = subreddit
        self.topics = topics
        self.max_posts = max_posts
        self.max_comments = max_comments

        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )

        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.posts_list = []
        self.comments_list = []
        self.posts_df = pd.DataFrame()
        self.comments_df = pd.DataFrame()
        self.filename = f"{self.subreddit_name}_data.xlsx"

        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def run(self):
        try:
            self._fetch_posts_and_comments()
            self.transform_data()
            self.load_to_database()
            self.save_to_excel()
        except Exception as e:
            print(f"\n🚨 CRASH: {e}")
            self.save_to_excel()
            self.save_backup_copy()
            raise e

    def _fetch_posts_and_comments(self):
        for index, topic in enumerate(self.topics, start=1):
            print(f"\n🎬 Scraping topic {index}/{len(self.topics)}: {topic}")
            search_results = list(self.reddit.subreddit(self.subreddit_name).search(topic, limit=self.max_posts))

            for post in tqdm(search_results, desc=f"[{topic[:25]}]", ncols=100):
                body = getattr(post, "selftext", None)
                if body == "" or body is None:
                    body = None

                self.posts_list.append({
                    "topic": topic,
                    "post_id": post.id,
                    "title": post.title,
                    "body": body,
                    "subreddit": post.subreddit.display_name,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "timestamp": datetime.datetime.utcfromtimestamp(post.created_utc),
                    "author": str(post.author),
                    "url": post.url
                })

                self._fetch_valid_comments(post, topic)
                time.sleep(1.5)

    def _fetch_valid_comments(self, post, topic):
        try:
            post.comments.replace_more(limit=0)
            comments = post.comments.list()

            collected = 0
            for comment in comments:
                if comment.body.lower() in ("[deleted]", "[removed]"):
                    continue

                self.comments_list.append({
                    "topic": topic,
                    "comment_id": comment.id,
                    "post_id": post.id,
                    "author": str(comment.author),
                    "body": comment.body,
                    "timestamp": datetime.datetime.utcfromtimestamp(comment.created_utc),
                    "comment_length": len(comment.body.split()),
                    "score": comment.score
                })

                collected += 1
                if collected >= self.max_comments:
                    break

        except Exception as e:
            print(f"❌ Error fetching comments from post {post.id}: {e}")
            time.sleep(5)

    def analyze_sentiment(self, text):
        if not text:
            return None
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores["compound"]
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
            return None

    def calculate_opinion_strength(self, text, polarity=None):
        if not text:
            return None
        try:
            blob = TextBlob(text)
            subjectivity = blob.sentiment.subjectivity
            polarity = polarity if polarity is not None else self.analyze_sentiment(text)
            return abs(polarity) * subjectivity if polarity is not None else None
        except Exception as e:
            print(f"⚠️ Opinion strength calculation failed: {e}")
            return None

    def calculate_realism_score(self, polarity, opinion_strength):
        if polarity is None or opinion_strength is None:
            return None
        return round((1 - abs(polarity)) * (1 - opinion_strength), 3)

    def calculate_named_entity_score(self, text):
        if not text:
            return 0.0
        doc = real_world_entity_parser(text.strip())
        entity_count = sum(1 for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT"})
        return round(min(entity_count / 5.0, 1.0), 3)

    def calculate_plausibility_score(self, text, polarity, opinion_strength):
        realism = self.calculate_realism_score(polarity, opinion_strength)
        named_entity_score = self.calculate_named_entity_score(text)
        if realism is None:
            return None
        plausibility = (
            0.6 * realism +
            0.15 * named_entity_score +
            0.15 * (1 - opinion_strength) +
            0.1 * (1 - abs(polarity))
        )
        return round(plausibility, 3)

    def transform_data(self):
        print("🔄 Running default transform_data: sentiment + opinion + plausibility")

        if self.comments_list:
            for comment in self.comments_list:
                body = comment.get("body")
                polarity = self.analyze_sentiment(body)
                opinion_strength = self.calculate_opinion_strength(body, polarity)
                plausibility_score = self.calculate_plausibility_score(body, polarity, opinion_strength)
                comment["sentiment_polarity"] = polarity
                comment["opinion_strength"] = opinion_strength
                comment["plausibility_score"] = plausibility_score

        if self.posts_list:
            for post in self.posts_list:
                body = post.get("body")
                polarity = self.analyze_sentiment(body)
                opinion_strength = self.calculate_opinion_strength(body, polarity)
                plausibility_score = self.calculate_plausibility_score(body, polarity, opinion_strength)
                post["sentiment_polarity"] = polarity
                post["opinion_strength"] = opinion_strength
                post["plausibility_score"] = plausibility_score

    def load_to_database(self, db_config=None):
        pass

    def save_to_excel(self, filename=None):
        if not os.path.exists("data"):
            os.makedirs("data")

        if not filename:
            base_filename = os.path.splitext(self.filename)[0]
            counter = 1
            filename = os.path.join("data", f"{base_filename}.xlsx")
            while os.path.exists(filename):
                filename = os.path.join("data", f"{base_filename}_{counter}.xlsx")
                counter += 1

        self.posts_df = pd.DataFrame(self.posts_list)
        self.comments_df = pd.DataFrame(self.comments_list)

        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            self.posts_df.to_excel(writer, index=False, sheet_name="Posts")
            self.comments_df.to_excel(writer, index=False, sheet_name="Comments")

        print(f"💾 Main file updated: {filename}")
        self.posts_list.clear()
        self.comments_list.clear()

    def save_backup_copy(self):
        if not os.path.exists("data"):
            os.makedirs("data")

        base_name = f"{self.subreddit_name}_crash_backup_{self._timestamp()}"
        filename = os.path.join("data", f"{base_name}.xlsx")

        counter = 1
        while os.path.exists(filename):
            filename = os.path.join("data", f"{base_name}_{counter}.xlsx")
            counter += 1

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.posts_df.to_excel(writer, index=False, sheet_name="Posts")
            self.comments_df.to_excel(writer, index=False, sheet_name="Comments")

        print(f"🛑 Crash backup saved as {filename}")

    def _timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
