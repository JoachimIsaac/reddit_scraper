# Global Sentiment Labeling Model 
# ----------------------------------------------------------
# Label = Positive     if polarity >  0.1
# Label = Negative     if polarity < -0.1
# Label = Neutral      if polarity between -0.1 and 0.1



import pytest
from black_mirror_scraper import BlackMirrorScraper  # adjust import path if needed
import logging

# Optional: if you don't use conftest.py
scraper = BlackMirrorScraper(topics=["test"])

# === SENTIMENT POLARITY TESTS ===

@pytest.mark.sentiment
def test_high_positive_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("I absolutely love this show! It's perfect.")
    logging.info(f"[High Positive] Got polarity = {polarity:.4f} | Expected: > 0.5")
    assert polarity > 0.5

@pytest.mark.sentiment
def test_high_negative_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("This was horrible. I hated every second.")
    logging.info(f"[High Negative] Got polarity = {polarity:.4f} | Expected: < -0.5")
    assert polarity < -0.5

@pytest.mark.sentiment
def test_neutral_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("The episode was released last year.")
    logging.info(f"[Neutral] Got polarity = {polarity:.4f} | Expected: between -0.2 and 0.2")
    assert -0.2 <= polarity <= 0.2

@pytest.mark.sentiment
def test_sarcastic_positive_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("Oh great, another disasterpiece. Just what we needed.")
    logging.info(f"[Sarcastic Positive] Got polarity = {polarity:.4f} | Expected: between 0.6 and 0.7 (VADER sarcasm baseline)")
    assert 0.6 <= polarity <= 0.7

@pytest.mark.sentiment
def test_sarcastic_negative_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("Wow, perfect plan! Ruin everything again!")
    logging.info(f"[Sarcastic Negative] Got polarity = {polarity:.4f} | Expected: between 0.5 and 0.7 (VADER misreads sarcasm)")
    assert 0.5 <= polarity <= 0.7

@pytest.mark.sentiment
def test_emoji_positive_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("This was amazing! 😍🔥💯")
    logging.info(f"[Emoji Positive] Got polarity = {polarity:.4f} | Expected: > 0.6")
    assert polarity > 0.6

@pytest.mark.sentiment
def test_emoji_negative_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("This episode sucked 😡🤬")
    logging.info(f"[Emoji Negative] Got polarity = {polarity:.4f} | Expected: between -0.55 and -0.4")
    assert -0.55 <= polarity <= -0.4

@pytest.mark.sentiment
def test_emoji_neutral_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("It aired last night. 📺")
    logging.info(f"[Emoji Neutral] Got polarity = {polarity:.4f} | Expected: between -0.3 and 0.3")
    assert -0.3 <= polarity <= 0.3

@pytest.mark.sentiment
def test_mixed_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("This episode was fun but the ending sucked.")
    logging.info(f"[Mixed] Got polarity = {polarity:.4f} | Expected: between -0.5 and 0.4")
    assert -0.5 <= polarity <= 0.4

@pytest.mark.sentiment
def test_question_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("Did anyone else think that was weird?")
    logging.info(f"[Question Sentiment] Got polarity = {polarity:.4f} | Expected: between -0.3 and 0.3")
    assert -0.3 <= polarity <= 0.3

@pytest.mark.sentiment
def test_shouting_negative_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("I HATED THIS EPISODE SO MUCH")
    logging.info(f"[Shouting Negative] Got polarity = {polarity:.4f} | Expected: < -0.3")
    assert polarity < -0.3

@pytest.mark.sentiment
def test_negation_handling():
    polarity, _ = scraper._analyze_text_sentiment("I didn’t like it.")
    logging.info(f"[Negation Handling] Got polarity = {polarity:.4f} | Expected: < 0.4")
    assert polarity < 0.4

@pytest.mark.sentiment
def test_double_negative_flips_positive():
    polarity, _ = scraper._analyze_text_sentiment("It wasn't that bad.")
    logging.info(f"[Double Negative] Got polarity = {polarity:.4f} | Expected: >= 0.0 (muted positive)")
    assert polarity >= 0.0

@pytest.mark.sentiment
def test_emoji_only_positive():
    polarity, _ = scraper._analyze_text_sentiment("😍🔥💯")
    logging.info(f"[Emoji-Only Positive] Got polarity = {polarity:.4f} | Expected: > 0.1")
    assert polarity > 0.1

@pytest.mark.sentiment
def test_emoji_only_negative():
    polarity, _ = scraper._analyze_text_sentiment("😡🤬💀")
    logging.info(f"[Emoji-Only Negative] Got polarity = {polarity:.4f} | Expected: <= 0.1")
    assert polarity <= 0.1
@pytest.mark.sentiment
def test_capslock_positive_sentiment():
    polarity, _ = scraper._analyze_text_sentiment("THIS WAS SO GOOD OMG")
    logging.info(f"[Capslock Positive] Got polarity = {polarity:.4f} | Expected: >= 0.4")
    assert polarity >= 0.4


# === OPINION STRENGTH TESTS ===

@pytest.mark.opinion
def test_strong_positive_opinion():
    _, strength = scraper._analyze_text_sentiment("I firmly believe this is the best episode ever made.")
    assert strength > 0.5

@pytest.mark.opinion
def test_strong_negative_opinion():
    _, strength = scraper._analyze_text_sentiment("This is absolutely the worst episode I've ever seen.")
    assert strength > 0.5

@pytest.mark.opinion
def test_weak_opinion():
    _, strength = scraper._analyze_text_sentiment("Maybe it's okay. Some people might like it.")
    assert 0.1 <= strength <= 0.4

@pytest.mark.opinion
def test_objective_fact_opinion_strength():
    _, strength = scraper._analyze_text_sentiment("The show has six seasons.")
    assert strength < 0.1

@pytest.mark.opinion
def test_uncertain_expression_opinion_strength():
    _, strength = scraper._analyze_text_sentiment("I’m not really sure what I think about this.")
    assert strength < 0.3

# === PLAUSIBILITY SCORE TESTS ===

@pytest.mark.plausibility
def test_realistic_blackmail_scenario():
    polarity, strength = scraper._analyze_text_sentiment("A politician was blackmailed using deepfake videos.")
    score = scraper.calculate_plausibility_score("A politician was blackmailed using deepfake videos.", polarity, strength)
    assert score > 0.6

@pytest.mark.plausibility
def test_fantasy_upload_to_cloud_after_death():
    polarity, strength = scraper._analyze_text_sentiment("After death, he uploaded his soul to the cloud and became immortal.")
    score = scraper.calculate_plausibility_score("After death, he uploaded his soul to the cloud and became immortal.", polarity, strength)
    assert score < 0.4

@pytest.mark.plausibility
def test_ambiguous_predictive_policing():
    polarity, strength = scraper._analyze_text_sentiment("The AI system knows your crimes before you commit them.")
    score = scraper.calculate_plausibility_score("The AI system knows your crimes before you commit them.", polarity, strength)
    assert 0.4 <= score <= 0.8

@pytest.mark.plausibility
def test_named_entities_but_unrealistic_context():
    text = "Elon Musk built a satellite that reads your memories and sells them."
    polarity, strength = scraper._analyze_text_sentiment(text)
    score = scraper.calculate_plausibility_score(text, polarity, strength)
    assert 0.3 <= score <= 0.7  # Good entities, but implausible concept

@pytest.mark.plausibility
def test_sarcastic_realistic_statement():
    text = "Oh sure, like politicians never lie — totally trustworthy folks."
    polarity, strength = scraper._analyze_text_sentiment(text)
    score = scraper.calculate_plausibility_score(text, polarity, strength)
    assert 0.3 <= score <= 0.7  # Language is sarcastic but grounded
