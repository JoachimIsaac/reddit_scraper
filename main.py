
from black_mirror_scraper import BlackMirrorScraper

episodes = [
    "The National Anthem", "Fifteen Million Merits", "The Entire History of You",
    "Be Right Back", "White Bear", "The Waldo Moment", "White Christmas",
    "Nosedive", "Playtest", "Shut Up and Dance", "San Junipero", "Men Against Fire", "Hated in the Nation",
    "USS Callister", "Arkangel", "Crocodile", "Hang the DJ", "Metalhead", "Black Museum",
    "Striking Vipers", "Smithereens", "Rachel, Jack and Ashley Too",
    "Joan Is Awful", "Loch Henry", "Beyond the Sea", "Mazey Day", "Demon 79",
    "Common People",       # S7‑1
    "Bête Noire",          # S7‑2
    "Hotel Reverie",       # S7‑3
    "Plaything",           # S7‑4
    "Eulogy",              # S7‑5
    "USS Callister: Into Infinity"  # S7‑6 (sequel to USS Callister)
]

if __name__ == "__main__":
    scraper = BlackMirrorScraper(topics=episodes, max_posts=5, max_comments=10)#5POST  -> 10COMMENT
    scraper.run()
    #print(scraper._emoji_sentiment_boost(" 😍💀😆🤬👍👎😊😡I HATED THIS 😡🤬 IT WAS AWFUL!!! 😍🔥💯😄😆 The visuals were cool 😍 but the story sucked 💩😊🥰👍👏🎉✨🌟 😡🤬👿💀💩👎😤😠😭😣😫😩 "))