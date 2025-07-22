import praw
import pandas as pd
import datetime
import os
import time
from openpyxl import load_workbook
from dotenv import load_dotenv
import warnings
from tqdm import tqdm

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
                self.posts_list.append({
                    "topic": topic,
                    "post_id": post.id,
                    "title": post.title,
                    "subreddit": post.subreddit.display_name,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.datetime.utcfromtimestamp(post.created_utc),
                    "author": str(post.author),
                    "url": post.url
                })

                self._fetch_valid_comments(post, topic)
                time.sleep(1.5)  # throttle API usage

            self.save_to_excel()  # autosave after each topic

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
                    "created_utc": datetime.datetime.utcfromtimestamp(comment.created_utc),
                    "word_count": len(comment.body.split()),
                    "score": comment.score
                })

                collected += 1
                if collected >= self.max_comments:
                    break

        except Exception as e:
            print(f"❌ Error fetching comments from post {post.id}: {e}")
            time.sleep(5)

    def apply_sentiment_analysis(self):
        """
        Placeholder method to apply sentiment analysis to self.comments_df
        """
        pass

    def calculate_realism_score(self):
        """
        Placeholder: meant for child classes (e.g., BlackMirrorScraper)
        """
        pass

    def load_to_database(self, db_config=None):
        """
        Placeholder for loading posts_df and comments_df into SQL DB
        """
        pass

    def save_to_excel(self, filename=None):
        if not os.path.exists("data"):
            os.makedirs("data")

        filename = filename or os.path.join("data", self.filename)
        df_posts = pd.DataFrame(self.posts_list)
        df_comments = pd.DataFrame(self.comments_list)

        if os.path.exists(filename):
            try:
                existing_posts = pd.read_excel(filename, sheet_name="Posts")
            except ValueError:
                existing_posts = pd.DataFrame()

            try:
                existing_comments = pd.read_excel(filename, sheet_name="Comments")
            except ValueError:
                existing_comments = pd.DataFrame()
        else:
            existing_posts = pd.DataFrame()
            existing_comments = pd.DataFrame()

        df_posts = pd.concat([existing_posts, df_posts], ignore_index=True)
        df_comments = pd.concat([existing_comments, df_comments], ignore_index=True)

        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            df_posts.to_excel(writer, index=False, sheet_name="Posts")
            df_comments.to_excel(writer, index=False, sheet_name="Comments")

        print(f"💾 Main file updated: {filename}")
        self.posts_list.clear()
        self.comments_list.clear()

    def save_backup_copy(self):
        if not os.path.exists("data"):
            os.makedirs("data")

        timestamped_name = os.path.join("data", f"{self.subreddit_name}_crash_backup_{self._timestamp()}.xlsx")

        with pd.ExcelWriter(timestamped_name, engine='openpyxl') as writer:
            self.posts_df.to_excel(writer, index=False, sheet_name="Posts")
            self.comments_df.to_excel(writer, index=False, sheet_name="Comments")

        print(f"🛑 Crash backup saved as {timestamped_name}")

    def _timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
