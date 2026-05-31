import pandas as pd

verse_counts = {
    "Romans": [32, 29, 31, 25, 21, 23, 25, 39, 33, 21, 36, 21, 14, 23, 33, 27],  # 16 chapters
    "1 Corinthians": [31, 16, 23, 21, 13, 20, 40, 13, 27, 33, 34, 31, 13, 40, 58, 24],  # 16 chapters
    "2 Corinthians": [24, 17, 18, 18, 21, 18, 16, 24, 15, 18, 33, 21, 14],  # 13 chapters
    "Galatians": [24, 21, 29, 31, 26, 18],  # 6 chapters
    "Philippians": [30, 30, 21, 23],  # 4 chapters
    "1 Thessalonians": [10, 20, 13, 18, 28],  # 5 chapters
    "2 Thessalonians": [12, 17, 18],  # 3 chapters
    "Philemon": [25],
    "Hebrews": [14, 18, 19, 16, 14, 20, 28, 13, 28, 39, 40, 29, 25],  # 13 chapters
    "4 Maccabees": [35, 24, 21, 26, 38, 35, 23, 29, 32, 21, 27, 19, 27, 20, 32, 25, 24, 24]  # 18 chapters
    }


# flatten all verses into rows
rows = []

for letter, chapters in verse_counts.items():
    for chap_num, num_verses in enumerate(chapters, start=1):
        for verse_num in range(1, num_verses + 1):
            rows.append({
                "Letter": letter,
                "Chapter": chap_num,
                "Verse": verse_num,
                "Segment": f"{chap_num}:{verse_num}"
            })

# create dataframe
verses_df = pd.DataFrame(rows)

# save to csv
verses_df.to_csv("output/verse_references_per_letter.csv", index=False)

# preview
print(verses_df.head())
