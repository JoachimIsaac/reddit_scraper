import pytest
from black_mirror_scraper import BlackMirrorScraper  # Adjust path if needed
import logging

# Setup
scraper = BlackMirrorScraper(topics=["test"])



# === SENTIMENT POLARITY TESTS ===

@pytest.mark.sentiment
def test_high_positive_sentiment():
    polarity = scraper.analyze_sentiment("I absolutely love this show! It's perfect.")
    logging.info(f"[High Positive] Got polarity = {polarity:.4f} | Expected: > 0.5")
    assert polarity > 0.5

@pytest.mark.sentiment
def test_high_negative_sentiment():
    polarity = scraper.analyze_sentiment("This was horrible. I hated every second.")
    logging.info(f"[High Negative] Got polarity = {polarity:.4f} | Expected: < -0.5")
    assert polarity < -0.5

@pytest.mark.sentiment
def test_neutral_sentiment():
    polarity = scraper.analyze_sentiment("The episode was released last year.")
    logging.info(f"[Neutral] Got polarity = {polarity:.4f} | Expected: between -0.2 and 0.2")
    assert -0.2 <= polarity <= 0.2

@pytest.mark.sentiment
def test_sarcastic_positive_sentiment():
    polarity = scraper.analyze_sentiment("Oh great, another disasterpiece. Just what we needed.")
    logging.info(f"[Sarcastic Positive] Got polarity = {polarity:.4f} | Expected: between 0.6 and 0.7")
    assert 0.6 <= polarity <= 0.7

@pytest.mark.sentiment
def test_sarcastic_negative_sentiment():
    polarity = scraper.analyze_sentiment("Wow, perfect plan! Ruin everything again!")
    logging.info(f"[Sarcastic Negative] Got polarity = {polarity:.4f} | Expected: between 0.5 and 0.7")
    assert 0.5 <= polarity <= 0.7

@pytest.mark.sentiment
def test_emoji_positive_sentiment():
    polarity = scraper.analyze_sentiment("This was amazing! 😍🔥💯")
    logging.info(f"[Emoji Positive] Got polarity = {polarity:.4f} | Expected: > 0.6")
    assert polarity > 0.6

@pytest.mark.sentiment
def test_emoji_negative_sentiment():
    polarity = scraper.analyze_sentiment("This episode sucked 😡🤬")
    logging.info(f"[Emoji Negative] Got polarity = {polarity:.4f} | Expected: between -0.55 and -0.4")
    assert -0.55 <= polarity <= -0.4

@pytest.mark.sentiment
def test_emoji_neutral_sentiment():
    polarity = scraper.analyze_sentiment("It aired last night. 📺")
    logging.info(f"[Emoji Neutral] Got polarity = {polarity:.4f} | Expected: between -0.3 and 0.3")
    assert -0.3 <= polarity <= 0.3

@pytest.mark.sentiment
def test_mixed_sentiment():
    polarity = scraper.analyze_sentiment("This episode was fun but the ending sucked.")
    logging.info(f"[Mixed] Got polarity = {polarity:.4f} | Expected: between -0.5 and 0.4")
    assert -0.5 <= polarity <= 0.4

@pytest.mark.sentiment
def test_question_sentiment():
    polarity = scraper.analyze_sentiment("Did anyone else think that was weird?")
    logging.info(f"[Question Sentiment] Got polarity = {polarity:.4f} | Expected: between -0.3 and 0.3")
    assert -0.3 <= polarity <= 0.3

@pytest.mark.sentiment
def test_shouting_negative_sentiment():
    polarity = scraper.analyze_sentiment("I HATED THIS EPISODE SO MUCH")
    logging.info(f"[Shouting Negative] Got polarity = {polarity:.4f} | Expected: < -0.3")
    assert polarity < -0.3

@pytest.mark.sentiment
def test_negation_handling():
    polarity = scraper.analyze_sentiment("I didn’t like it.")
    logging.info(f"[Negation Handling] Got polarity = {polarity:.4f} | Expected: < 0.4")
    assert polarity < 0.4

@pytest.mark.sentiment
def test_double_negative_flips_positive():
    polarity = scraper.analyze_sentiment("It wasn't that bad.")
    logging.info(f"[Double Negative] Got polarity = {polarity:.4f} | Expected: >= 0.0")
    assert polarity >= 0.0

@pytest.mark.sentiment
def test_emoji_only_positive():
    polarity = scraper.analyze_sentiment("😍🔥💯")
    logging.info(f"[Emoji-Only Positive] Got polarity = {polarity:.4f} | Expected: > 0.1")
    assert polarity > 0.1

@pytest.mark.sentiment
def test_emoji_only_negative():
    polarity = scraper.analyze_sentiment("😡🤬💀")
    logging.info(f"[Emoji-Only Negative] Got polarity = {polarity:.4f} | Expected: <= 0.1")
    assert polarity <= 0.1

@pytest.mark.sentiment
def test_capslock_positive_sentiment():
    polarity = scraper.analyze_sentiment("THIS WAS SO GOOD OMG")
    logging.info(f"[Capslock Positive] Got polarity = {polarity:.4f} | Expected: >= 0.4")
    assert polarity >= 0.4



# === OPINION STRENGTH TESTS ===

@pytest.mark.opinion
def test_strong_positive_opinion():
    text = "I firmly believe this is the best episode ever made."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    logging.info(f"[Strong Positive] Got strength = {strength:.4f} | Expected: > 0.5")
    assert strength > 0.5

@pytest.mark.opinion
def test_strong_negative_opinion():
    text = "This is absolutely the worst episode I've ever seen."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    logging.info(f"[Strong Negative] Got strength = {strength:.4f} | Expected: > 0.5")
    assert strength > 0.5

@pytest.mark.opinion
def test_weak_opinion():
    text = "Maybe it's okay. Some people might like it."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    logging.info(f"[Weak Opinion] Got strength = {strength:.4f} | Expected: 0.1 <= strength <= 0.4")
    assert 0.1 <= strength <= 0.4

@pytest.mark.opinion
def test_objective_fact_opinion_strength():
    text = "The show has six seasons."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    logging.info(f"[Objective Fact] Got strength = {strength:.4f} | Expected: < 0.1")
    assert strength < 0.1

@pytest.mark.opinion
def test_uncertain_expression_opinion_strength():
    text = "I’m not really sure what I think about this."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    logging.info(f"[Uncertain Expression] Got strength = {strength:.4f} | Expected: < 0.3")
    assert strength < 0.3



# === PLAUSIBILITY SCORE TESTS ===

@pytest.mark.plausibility
def test_realistic_blackmail_scenario():
    text = "A politician was blackmailed using deepfake videos."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    score = scraper.calculate_plausibility_score(text, polarity, strength)
    assert score > 0.6

@pytest.mark.plausibility
def test_fantasy_upload_to_cloud_after_death():
    text = "After death, he uploaded his soul to the cloud and became immortal."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    score = scraper.calculate_plausibility_score(text, polarity, strength)
    assert score < 0.4

@pytest.mark.plausibility
def test_ambiguous_predictive_policing():
    text = "The AI system knows your crimes before you commit them."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    score = scraper.calculate_plausibility_score(text, polarity, strength)
    assert 0.4 <= score <= 0.8

@pytest.mark.plausibility
def test_named_entities_but_unrealistic_context():
    text = "Elon Musk built a satellite that reads your memories and sells them."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    score = scraper.calculate_plausibility_score(text, polarity, strength)
    assert 0.3 <= score <= 0.7

@pytest.mark.plausibility
def test_sarcastic_realistic_statement():
    text = "Oh sure, like politicians never lie — totally trustworthy folks."
    polarity = scraper.analyze_sentiment(text)
    strength = scraper.calculate_opinion_strength(text, polarity)
    score = scraper.calculate_plausibility_score(text, polarity, strength)
    assert 0.3 <= score <= 0.7
