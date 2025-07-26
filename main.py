
from black_mirror_scraper import BlackMirrorScraper

episodes = [
    "The National Anthem", "Fifteen Million Merits", "The Entire History of You",
    "Be Right Back", "White Bear", "The Waldo Moment", "White Christmas",
    "Nosedive", "Playtest", "Shut Up and Dance", "San Junipero", "Men Against Fire", "Hated in the Nation",
    "USS Callister", "Arkangel", "Crocodile", "Hang the DJ", "Metalhead", "Black Museum",
    "Striking Vipers", "Smithereens", "Rachel, Jack and Ashley Too",
    "Joan Is Awful", "Loch Henry", "Beyond the Sea", "Mazey Day", "Demon 79"
]

if __name__ == "__main__":
    scraper = BlackMirrorScraper(topics=episodes, max_posts=2, max_comments=2)
    scraper.run()
    #scraper.run_quality_tests()