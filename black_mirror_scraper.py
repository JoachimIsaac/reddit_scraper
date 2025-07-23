from reddit_scraper import RedditScraper

class BlackMirrorScraper(RedditScraper):
    def __init__(self, topics, max_posts=100, max_comments=50):
        super().__init__(subreddit="blackmirror", topics=topics, max_posts=max_posts, max_comments=max_comments)

    # def calculate_realism_score(self):
    #     """
    #     Placeholder for realism score logic specific to Black Mirror episodes.
    #     """
    #     pass

    """
    I need to seperate the realism logic and opinion strength logic from the main class and keep it in black mirror child class 
    because they are unqiue to black mirror . essentially i have to keep building from the black mirror class the new transfomation 
    methods and not the parent class. 
    """

    def run(self):
        try:
            self._fetch_posts_and_comments()
            self.transform_data()
            self.load_to_database()
            self.save_to_excel()
        except Exception as e:
            print(f"\n🚨 CRASH in BlackMirrorScraper: {e}")
            self.save_to_excel()
            self.save_backup_copy()
            raise e
