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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer, BOOSTER_DICT
from textblob import TextBlob
import re as regex

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

        self.booster_dict = BOOSTER_DICT
        # Comprehensive list of hedging words that indicate uncertainty, doubt, or qualification
        self.hedging_words = {
            # Basic uncertainty
            'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'would',
            'may', 'can', 'should', 'ought', 'must', 'shall',
            
            # Appearances and impressions
            'seems', 'appears', 'looks like', 'sounds like', 'feels like',
            'strikes me as', 'comes across as', 'gives the impression',
            
            # Approximations and qualifiers
            'sort of', 'kind of', 'somewhat', 'rather', 'quite', 'fairly',
            'relatively', 'approximately', 'roughly', 'about', 'around',
            'nearly', 'almost', 'basically', 'essentially', 'virtually',
            'practically', 'more or less', 'in a way', 'in some ways',
            
            # Frequency and typicality
            'generally', 'usually', 'typically', 'normally', 'ordinarily',
            'commonly', 'frequently', 'often', 'sometimes', 'occasionally',
            'rarely', 'seldom', 'hardly ever', 'almost never',
            
            # Reported information
            'supposedly', 'allegedly', 'reportedly', 'apparently', 'ostensibly',
            'purportedly', 'rumored', 'said to be', 'claimed to be',
            'believed to be', 'thought to be', 'considered to be',
            
            # Assumptions and expectations
            'presumably', 'assumably', 'presumably', 'presumably',
            'expected to', 'likely to', 'supposed to', 'meant to',
            'intended to', 'designed to', 'planned to',
            
            # Doubt and skepticism
            'doubt', 'doubtful', 'questionable', 'uncertain', 'unclear',
            'ambiguous', 'vague', 'unclear', 'unsettled', 'debatable',
            'controversial', 'disputed', 'contested', 'arguable',
            
            # Conditional language
            'if', 'assuming', 'provided that', 'given that', 'in case',
            'contingent on', 'dependent on', 'subject to', 'conditional on',
            
            # Softeners and mitigators
            'a bit', 'a little', 'slightly', 'marginally', 'minimally',
            'barely', 'scarcely', 'hardly', 'just', 'merely', 'simply',
            'only', 'merely', 'simply', 'just', 'purely', 'solely',
            
            # Comparative uncertainty
            'more or less', 'better or worse', 'sooner or later',
            'one way or another', 'for better or worse',
            
            # Temporal uncertainty
            'eventually', 'ultimately', 'finally', 'in the end',
            'sooner or later', 'one day', 'someday', 'at some point',
            
            # Spatial uncertainty
            'somewhere', 'someplace', 'somewhere around', 'in the area',
            'in the vicinity', 'nearby', 'close to', 'not far from',
            
            # Quantity uncertainty
            'some', 'several', 'a few', 'a couple', 'a handful',
            'various', 'varying', 'different', 'diverse', 'assorted',
            'mixed', 'varied', 'multiple', 'numerous', 'many',
            
            # Quality uncertainty
            'decent', 'reasonable', 'acceptable', 'adequate', 'satisfactory',
            'passable', 'tolerable', 'bearable', 'manageable', 'workable',
            
            # Intensity modifiers
            'somewhat', 'partially', 'in part', 'to some extent',
            'to a degree', 'in a sense', 'in some sense', 'in a manner',
            'in a fashion', 'in a way', 'in some way',
            
            # Expert opinion qualifiers
            'according to', 'based on', 'per', 'as per', 'in accordance with',
            'in line with', 'consistent with', 'in keeping with',
            
            # Personal opinion markers
            'i think', 'i believe', 'i feel', 'i guess', 'i suppose',
            'i assume', 'i imagine', 'i reckon', 'i figure', 'i gather',
            'i understand', 'i hear', 'i see', 'i notice',
            
            # Evidence qualifiers
            'apparently', 'evidently', 'obviously', 'clearly', 'plainly',
            'manifestly', 'patently', 'undoubtedly', 'indisputably',
            'unquestionably', 'definitely', 'certainly', 'surely',
            
            # Time-based uncertainty
            'recently', 'lately', 'nowadays', 'these days', 'currently',
            'presently', 'at present', 'at the moment', 'right now',
            'for now', 'for the time being', 'temporarily',
            
            # Method uncertainty
            'somehow', 'someway', 'in some way', 'by some means',
            'through some process', 'via some method', 'using some approach',
            
            # Result uncertainty
            'hopefully', 'ideally', 'preferably', 'ideally', 'optimally',
            'best case', 'worst case', 'in theory', 'in practice',
            'in reality', 'in actuality', 'in fact', 'in truth',
            
            # Comparative uncertainty
            'more or less', 'better or worse', 'sooner or later',
            'one way or another', 'for better or worse', 'like it or not',
            
            # Conditional uncertainty
            'depending on', 'subject to', 'contingent upon', 'based on',
            'assuming', 'provided that', 'given that', 'if', 'when',
            'unless', 'except', 'barring', 'failing', 'short of',
            
            # Degree uncertainty
            'to some degree', 'to some extent', 'in part', 'partially',
            'somewhat', 'rather', 'quite', 'fairly', 'reasonably',
            'moderately', 'adequately', 'sufficiently', 'appropriately',
            
            # Source uncertainty
            'allegedly', 'supposedly', 'reportedly', 'apparently',
            'ostensibly', 'purportedly', 'rumored', 'said to be',
            'claimed to be', 'believed to be', 'thought to be',
            
            # Process uncertainty
            'somehow', 'someway', 'in some way', 'by some means',
            'through some process', 'via some method', 'using some approach',
            'in some manner', 'in some fashion', 'in some respect',
            
            # Outcome uncertainty
            'hopefully', 'ideally', 'preferably', 'optimally',
            'best case', 'worst case', 'in theory', 'in practice',
            'in reality', 'in actuality', 'in fact', 'in truth',
            'actually', 'really', 'truly', 'genuinely', 'honestly'
        }

   
        self.emoji_df = pd.read_csv("emoji_sentiment_data.csv")
        self.positive_emojis = set(self.emoji_df[self.emoji_df['sentiment score'] > 0]['Emoji'])
        self.negative_emojis = set(self.emoji_df[self.emoji_df['sentiment score'] < 0]['Emoji'])

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
            base_strength = abs(polarity) * subjectivity

            certainty_score = self._certainty_word_boost(text)
            hedge_penalty = self._hedging_penalty(text)
            emphasis_boost = self._text_emphasis_boost(text)
            emoji_boost = self._emoji_sentiment_boost(text)

            text_lower = text.lower()
            normalized = text_lower.replace("’", "'").strip()

            # Boost if polarity is strong
            if abs(polarity) > 0.6:
                base_strength += 0.1

            # Pattern-based: certainty verb + superlative adjective
            certainty_verbs = ["believe", "know", "guarantee", "stand by", "swear", "firmly think", "can confirm"]
            superlatives = ["best", "worst", "greatest", "most amazing", "least enjoyable", "biggest", "craziest"]
            if any(v in text_lower for v in certainty_verbs) and any(s in text_lower for s in superlatives):
                base_strength += 0.1

            # Soft negation pattern (expanded)
            negated_verbs = [
                "like", "love", "enjoy", "recommend", "prefer",
                "appreciate", "stand", "tolerate", "hate", "support"
            ]
            if polarity == 0 and any(f"didn't {v}" in normalized for v in negated_verbs):
                base_strength += 0.3

            if polarity == 0 and any(f"wasn't {adj}" in normalized for adj in ["great", "amazing", "terrible", "bad", "funny", "interesting"]):
                base_strength += 0.2

            # Mixed opinion clause handling
            if "but" in text_lower or "however" in text_lower:
                base_strength += 0.15

            strength = base_strength * (1 + certainty_score + emoji_boost + emphasis_boost - hedge_penalty)
            return min(max(strength, 0.0), 1.0)

        except Exception as e:
            print(f"⚠️ Opinion strength calculation failed: {e}")
            return None
        

    def _certainty_word_boost(self, text):
        words = text.lower().split()
        return 0.15 * sum(1 for word in words if word in self.booster_dict)

    def _hedging_penalty(self, text):
        text_lower = text.lower()
        return 0.1 * sum(1 for word in self.hedging_words if word in text_lower)

    def _text_emphasis_boost(self, text):
        boost = 0.0
        if text.isupper():
            boost += 0.2  # increased from 0.1
        if "!" in text:
            boost += 0.2  # increased from 0.1
        return boost


    def _emoji_sentiment_boost(self, text):
        text = text.strip()
        positive_hits = sum(e in text for e in self.positive_emojis)
        negative_hits = sum(e in text for e in self.negative_emojis)
        print("Positive emoji hits:", [e for e in self.positive_emojis if e in text])
        print("Negative emoji hits:", [e for e in self.negative_emojis if e in text])

        boost = 0.1 * positive_hits - 0.1 * negative_hits

        # Refined fallback logic
        if boost == 0:
            if negative_hits > 0:
                boost -= 0.3
            elif positive_hits > 0:
                boost += 0.3

        return boost

    

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
    

    def calculate_plausibility_score_v2(self, text, polarity=None, strength=None):
        text = text.lower().strip()

        # Expanded keyword banks
        real_world_tech = {
            "ai", "artificial intelligence", "machine learning", "neuralink", "elon musk",
            "facebook", "meta", "google", "deepfake", "facial recognition", "data mining",
            "surveillance", "tiktok", "social credit", "credit score", "china", "government",
            "algorithm", "privacy breach", "drones", "apple", "iphone", "gps", "blockchain",
            "neural implants", "neural implant", "rating system", "data tracking", "metadata",
            "facial scanner", "data collection", "implant", "chip", "biometrics"
        }

        logic_indicators = {
            "could happen", "can happen", "might happen", "i can see this", "already happening",
            "already doing", "they're doing", "literally doing", "is literally what", "definitely happening",
            "likely in the future", "makes sense", "because", "due to", "as a result", "if this continues",
            "this is happening", "not far off", "not far from reality", "i wouldn't be surprised",
            "they're building it", "this is basically", "this could totally happen", "totally possible",
            "feels like", "the future is", "could actually happen", "we’ll all have", "this episode nailed it"
        }

        fantasy_flags = {
            "aliens", "soul", "telepathy", "implanted memories", "clone wars",
            "the moon is alive", "reptilian", "ghost", "afterlife", "simulation",
            "psychic", "mind reading", "time traveler", "resurrected", "immortal",
            "upload soul", "parallel universe", "alternate dimension"
        }

        # Scores
        tech_score = sum(1 for word in real_world_tech if word in text)
        logic_score = sum(1 for phrase in logic_indicators if phrase in text)
        fantasy_penalty = sum(1 for word in fantasy_flags if word in text)

        # Entity score using spaCy
        doc = real_world_entity_parser(text)
        entity_score = sum(1 for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT"})
        entity_score = min(entity_score, 3)

        # Proper noun signal (adds a tiny nudge)
        proper_noun_boost = 0.05 if sum(1 for token in doc if token.pos_ == "PROPN") >= 2 else 0

        # Real-world actor bonus (slightly increases plausibility)
        if any(x in text for x in ["elon", "bezos", "zuckerberg", "nsa", "cia", "facebook", "apple"]):
            real_actor_bonus = 0.1
        else:
            real_actor_bonus = 0

        # Sarcasm-based fallback
        sarcasm_bonus = 0
        if polarity is not None and strength is not None:
            if polarity < 0 and strength > 0.4:
                if any(x in text for x in ["sure", "totally", "never lie", "obviously not corrupt", "trustworthy folks", "oh sure", "as if"]):
                    sarcasm_bonus += 0.2

        # Soft realism cues (boost for generic plausible phrasing)
        soft_realism = 0
        if any(x in text for x in ["not sure how realistic", "seems real", "imagine if", "hits hard", "fiction but", "wild but true", "already a thing", "happening now"]):
            soft_realism += 0.2

        # Final score
        score = (
            0.3 * tech_score +
            0.35 * logic_score +
            0.25 * entity_score -
            0.4 * fantasy_penalty +
            sarcasm_bonus +
            soft_realism +
            proper_noun_boost +
            real_actor_bonus
        )

        # Floor bump for “engaged” speech even if no features triggered
        if score == 0 and polarity and strength:
            score += 0.1

        print(f"📊 TECH: {tech_score}, LOGIC: {logic_score}, ENT: {entity_score}, FANTASY: {fantasy_penalty}, SARCASM: {sarcasm_bonus}, SOFT: {soft_realism}, PROPN: {proper_noun_boost}, REAL: {real_actor_bonus}")
        print(f"🔎 FINAL: {score}")

        return max(0, min(round(score, 2), 5))
    


    def transform_data(self):
        print("🔄 Running default transform_data: sentiment + opinion + plausibility")

        if self.comments_list:
            for comment in self.comments_list:
                body = comment.get("body")
                polarity = self.analyze_sentiment(body)
                opinion_strength = self.calculate_opinion_strength(body, polarity)
                plausibility_score = self.calculate_plausibility_score_v2(body, polarity, opinion_strength)
                comment["sentiment_polarity"] = polarity
                comment["opinion_strength"] = opinion_strength
                comment["plausibility_score"] = plausibility_score

        if self.posts_list:
            for post in self.posts_list:
                body = post.get("body")
                polarity = self.analyze_sentiment(body)
                opinion_strength = self.calculate_opinion_strength(body, polarity)
                plausibility_score = self.calculate_plausibility_score_v2(body, polarity, opinion_strength)
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
