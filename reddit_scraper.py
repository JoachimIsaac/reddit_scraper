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
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



load_dotenv()

# Load spaCy model once
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
        """Returns VADER compound polarity and opinion strength based on TextBlob + VADER"""
        if not text:
            return None, None
        try:
            blob = TextBlob(text)
            vader_scores = self.vader_analyzer.polarity_scores(text)

            sentiment_polarity = vader_scores["compound"]
            textblob_subjectivity = blob.sentiment.subjectivity
            opinion_strength = abs(sentiment_polarity) * textblob_subjectivity

            return sentiment_polarity, opinion_strength
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
            return None, None

    def calculate_named_entity_score(self, text):
        """Scores how grounded a text is by checking real-world references."""
        if not text:
            return 0.0
        text = text.strip()# preserve casing to improve entity detection
        doc = real_world_entity_parser(text)
        entity_count = sum(1 for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT"})
        return round(min(entity_count / 5.0, 1.0), 3)  # Normalized to max 1.0

    def calculate_realism_score(self, polarity, opinion_strength):
        """Measures emotional neutrality and objectivity — realism is higher when polarity and emotion are lower."""
        if polarity is None or opinion_strength is None:
            return None
        return round((1 - abs(polarity)) * (1 - opinion_strength), 3)

    def calculate_plausibility_score(self, text, polarity, opinion_strength):
        """Combines realism, named entities, polarity, and emotional strength into a plausibility metric."""
        realism = self.calculate_realism_score(polarity, opinion_strength)
        named_entity_score = self.calculate_named_entity_score(text)
        if realism is None:
            return None
        plausibility = (
            0.6 * realism +
            0.15 * named_entity_score +
            0.15 * (1 - opinion_strength) +  # Emphasize groundedness
            0.1 * (1 - abs(polarity))        # Lower polarity = higher plausibility
        )
        return round(plausibility, 3)


    def transform_data(self):#may need to name this something else like transform data 
        if self.comments_list:
            print("🧠 Analyzing comment sentiment...")
            for comment in self.comments_list:
                body = comment.get("body")
                sentiment_polarity, opinion_strength = self._analyze_text_sentiment(body)
                comment["sentiment_polarity"] = sentiment_polarity
                comment["opinion_strength"] = opinion_strength
                comment["plausibility_score"] = self.calculate_plausibility_score(body, sentiment_polarity, opinion_strength)

        if self.posts_list:
            print("🧠 Analyzing post sentiment...")
            for post in self.posts_list:
                body = post.get("body")
                sentiment_polarity, opinion_strength = self._analyze_text_sentiment(body)
                post["sentiment_polarity"] = sentiment_polarity
                post["opinion_strength"] = opinion_strength
                post["plausibility_score"] = self.calculate_plausibility_score(body, sentiment_polarity, opinion_strength)


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
