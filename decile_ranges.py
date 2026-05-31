import pandas as pd

# load verse reference spreadsheet
verse_df = pd.read_csv("output/verse_references_per_letter.csv")

# function to assign deciles to each verse within a letter
def assign_deciles(df):
    decile_data = []

    # go letter by letter
    for letter, group in df.groupby("Letter"):
        group = group.reset_index(drop=True)
        total = len(group)
        base = total // 10
        remainder = total % 10

        # create decile bin sizes
        decile_sizes = [base + 1 if i < remainder else base for i in range(10)]

        # assign decile labels
        decile_labels = []
        for i, size in enumerate(decile_sizes):
            decile_labels.extend([i + 1] * size)

        group["Decile"] = decile_labels
        decile_data.append(group)

    # combine all letters back into one dataframe
    return pd.concat(decile_data, ignore_index=True)


# apply the decile assignment
verse_decile_df = assign_deciles(verse_df)

# save to a new csv file
verse_decile_df.to_csv("output/verse_references_with_deciles.csv", index=False)

# preview a few rows
print(verse_decile_df.head())

# load the decile-tagged verse reference data
df = pd.read_csv("output/verse_references_with_deciles.csv")

# create a summary of decile ranges
ranges = []

# group by letter and decile
for (letter, decile), group in df.groupby(["Letter", "Decile"]):
    start_ref = group.iloc[0]["Segment"]
    end_ref = group.iloc[-1]["Segment"]
    ranges.append({
        "Letter": letter,
        "Decile": decile,
        "Verse Range": f"{start_ref} – {end_ref}"
    })

# convert to dataframe
ranges_df = pd.DataFrame(ranges)

# sort for readability
ranges_df = ranges_df.sort_values(by=["Letter", "Decile"]).reset_index(drop=True)

# save to csv
ranges_df.to_csv("output/decile_verse_ranges.csv", index=False)

# preview
print(ranges_df.head(10))
