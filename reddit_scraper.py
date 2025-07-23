import praw
import pandas as pd
import datetime
import os
import time
from openpyxl import load_workbook
from dotenv import load_dotenv
import warnings
from tqdm import tqdm
from textblob import TextBlob

load_dotenv()

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

        self.posts_list = []
        self.comments_list = []
        self.posts_df = pd.DataFrame()
        self.comments_df = pd.DataFrame()
        self.filename = f"{self.subreddit_name}_data.xlsx"

        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def run(self):
        try:
            self._fetch_posts_and_comments()
            self.apply_sentiment_analysis()
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
                time.sleep(1.5)  # throttle API usage

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

    def _analyze_text_sentiment(self, text):
        """Returns sentiment polarity and subjectivity (converted to objectivity)."""
        if not text:
            return None, None
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            return polarity, 1 - subjectivity
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
            return None, None

    def apply_sentiment_analysis(self):
        if self.comments_list:
            print("🧠 Analyzing comment sentiment...")
            for comment in self.comments_list:
                polarity, objectivity = self._analyze_text_sentiment(comment.get("body"))
                comment["sentiment_polarity"] = polarity
                comment["objectivity_score"] = objectivity

        if self.posts_list:
            print("🧠 Analyzing post sentiment...")
            for post in self.posts_list:
                polarity, objectivity = self._analyze_text_sentiment(post.get("body"))
                post["sentiment_polarity"] = polarity
                post["objectivity_score"] = objectivity

    def calculate_realism_score(self):
        pass  # Placeholder for future realism logic

    def load_to_database(self, db_config=None):
        pass  # Placeholder for future SQL integration

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

        df_posts = self.posts_df
        df_comments = self.comments_df

        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            df_posts.to_excel(writer, index=False, sheet_name="Posts")
            df_comments.to_excel(writer, index=False, sheet_name="Comments")

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
