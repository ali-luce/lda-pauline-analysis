import pandas as pd
import os

# directories
clean_data_dir = "clean_data"
output_dir = "deciles"
os.makedirs(output_dir, exist_ok=True)

# assign verse-based deciles
def assign_biblical_deciles(verse_file):
    verse_df = pd.read_csv(verse_file)
    all_dfs = []

    for letter, group in verse_df.groupby("Letter"):
        group = group.reset_index(drop=True)
        total = len(group)
        base = total // 10
        remainder = total % 10
        decile_sizes = [base + 1 if i < remainder else base for i in range(10)]
        decile_labels = [d+1 for d, size in enumerate(decile_sizes) for _ in range(size)]
        group["Decile"] = decile_labels
        all_dfs.append(group)

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv("output/verse_references_with_deciles.csv", index=False)

    # save verse ranges
    ranges = []
    for (letter, decile), grp in full.groupby(["Letter", "Decile"]):
        start = grp.iloc[0]["Segment"]
        end = grp.iloc[-1]["Segment"]
        ranges.append({"Letter": letter, "Decile": decile, "Verse Range": f"{start} – {end}"})
    pd.DataFrame(ranges).to_csv("output/decile_verse_ranges.csv", index=False)

    return full


# assign deciles based on full structure list (not just features)
def assign_row_deciles_structural(input_path, output_path, text_name, section_list):
    df = pd.read_csv(input_path)

    # sort the full structure
    def segment_sort_key(seg):
        try:
            return tuple(int(p) for p in seg.split("."))
        except:
            return (float("inf"),)

    sorted_structure = sorted(section_list, key=segment_sort_key)

    # divide full structure into deciles
    total = len(sorted_structure)
    base, rem = total // 10, total % 10
    bin_sizes = [base + 1 if i < rem else base for i in range(10)]

    decile_map = {}
    idx = 0
    for decile, count in enumerate(bin_sizes, start=1):
        for _ in range(count):
            decile_map[sorted_structure[idx]] = int(decile)
            idx += 1

    # assign features to deciles based on structure map
    rows = []
    for _, row in df.iterrows():
        feat = row["Characteristics"].strip()
        if pd.isna(row["Appearances"]): continue
        for seg in row["Appearances"].split(","):
            seg = seg.strip()
            rows.append({
                "Text": text_name,
                "Characteristic": feat,
                "Segment": seg,
                "Decile": decile_map.get(seg)
            })

    final_df = pd.DataFrame(rows)
    final_df["Decile"] = final_df["Decile"].astype("Int64")  # ensures integers but allows for NaN
    final_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(final_df)} rows)")


# assign row-based deciles for non-biblical texts
def padded_key(seg):
    return [p.zfill(3) if p.isdigit() else p for p in seg.split(".")]

def assign_row_deciles(input_path, output_path, text_name):
    df = pd.read_csv(input_path)
    all_segments = set()
    for val in df["Appearances"].dropna():
        all_segments.update([v.strip() for v in val.split(",")])

    # sort those segments based on their numeric structure (e.g., 3.1.1 < 3.2.1)
    def segment_sort_key(seg):
        try:
            return tuple(int(p) for p in seg.split("."))
        except:
            return (float("inf"),)

    sorted_segments = sorted(all_segments, key=segment_sort_key)

    total = len(sorted_segments)
    base, rem = total // 10, total % 10
    bin_sizes = [base + 1 if i < rem else base for i in range(10)]

    # build a map of segment --> decile
    decile_map = {}
    idx = 0
    for decile, count in enumerate(bin_sizes, start=1):
        for _ in range(count):
            decile_map[sorted_segments[idx]] = decile
            idx += 1

    rows = []
    for _, row in df.iterrows():
        feat = row["Characteristics"].strip()
        if pd.isna(row["Appearances"]): continue
        for seg in row["Appearances"].split(","):
            seg = seg.strip()
            rows.append({
                "Text": text_name,
                "Characteristic": feat,
                "Segment": seg,
                "Decile": decile_map.get(seg)
            })

    out = pd.DataFrame(rows)
    out["Decile"] = out["Decile"].astype("Int64")
    out.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(out)} rows)")

# expand and merge biblical texts
def explode_and_merge_biblical(feature_file, deciles_df, letter):
    features = pd.read_csv(feature_file)
    exploded = []

    for _, row in features.iterrows():
        char = row["Characteristics"].strip()
        if pd.isna(row["Appearances"]): continue
        for ref in row["Appearances"].split(","):
            ref = ref.strip()
            if "." in ref:
                parts = ref.split(".")
                if len(parts) >= 2:
                    c, v = parts[0], parts[1]
                    exploded.append({
                        "Letter": letter,
                        "Characteristic": char,
                        "Segment": f"{c}:{v}"
                    })
                exploded.append({
                    "Letter": letter,
                    "Characteristic": char,
                    "Segment": f"{c}:{v}"

                })

    flat = pd.DataFrame(exploded)
    merged = pd.merge(flat, deciles_df[["Letter", "Segment", "Decile"]],
                      on=["Letter", "Segment"], how="left")
    out_path = os.path.join(output_dir, f"cleaned_{letter.lower()}_with_deciles.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(merged)} rows)")


# generate verse-based deciles and ranges
biblical_deciles = assign_biblical_deciles("output/verse_references_per_letter.csv")

# normalize non-pauline texts by row count
non_biblical_files = [
    "cleaned_4maccabees.csv",
    "cleaned_aelius_aristides_panathenaicus.csv",
    "cleaned_damascus_document.csv",
    "cleaned_epictetus_discourses.csv",
    "cleaned_philodemus_on_death.csv",
    "cleaned_philodemus_on_piety.csv",
    "cleaned_seneca_natural_questions.csv"
]

# helper function to expand segment ranges like "3.1.1-3.1.18" into a list of segments
def expand_ranges(range_list):
    segments = []
    for item in range_list:
        start, end = item.split("-")
        b1, c1, s1 = map(int, start.strip().split("."))
        b2, c2, s2 = map(int, end.strip().split("."))

        assert (b1, c1) == (b2, c2), "Start and end must be in the same chapter"

        for i in range(s1, s2 + 1):
            segments.append(f"{b1}.{c1}.{i}")
    return segments


# full structure list for seneca with preface treated as 3.1 and all others shifted
seneca_ranges = [
    "3.1.1-3.1.18", "3.2.1-3.2.2", "3.3.1-3.3.2", "3.4.1-3.4.1", "3.5.1-3.5.1",
    "3.6.1-3.6.1", "3.7.1-3.7.2", "3.8.1-3.8.4", "3.9.1-3.9.1",
    "3.10.1-3.10.3", "3.11.1-3.11.5", "3.12.1-3.12.6", "3.13.1-3.13.3",
    "3.14.1-3.14.2", "3.15.1-3.15.3", "3.16.1-3.16.8", "3.17.1-3.17.5",
    "3.18.1-3.18.3", "3.19.1-3.19.7", "3.20.1-3.20.4", "3.21.1-3.21.6",
    "3.22.1-3.22.2", "3.23.1-3.23.1", "3.24.1-3.24.1", "3.25.1-3.25.4",
    "3.26.1-3.26.12", "3.27.1-3.27.8", "3.28.1-3.28.15", "3.29.1-3.29.7",
    "3.30.1-3.30.9", "3.31.1-3.31.8"
]

# structural ranges for epictetus
epictetus_ranges = [
    "1.1.1-1.1.32", "1.2.1-1.2.37", "1.3.1-1.3.9", "1.4.1-1.4.32", "1.5.1-1.5.10",
    "1.6.1-1.6.43", "1.7.1-1.7.33", "1.8.1-1.8.16", "1.9.1-1.9.34", "1.10.1-1.10.13",
    "1.11.1-1.11.40", "1.12.1-1.12.35", "1.13.1-1.13.5", "1.14.1-1.14.17", "1.15.1-1.15.8",
    "1.16.1-1.16.21", "1.17.1-1.17.29", "1.18.1-1.18.23", "1.19.1-1.19.29", "1.20.1-1.20.19",
    "1.21.1-1.21.4", "1.22.1-1.22.21", "1.23.1-1.23.10", "1.24.1-1.24.20", "1.25.1-1.25.33",
    "1.26.1-1.26.18", "1.27.1-1.27.21", "1.28.1-1.28.33", "1.29.1-1.29.66", "1.30.1-1.30.7",
    "2.1.1-2.1.40", "2.2.1-2.2.26", "2.3.1-2.3.5", "2.4.1-2.4.11", "2.5.1-2.5.29",
    "2.6.1-2.6.27", "2.7.1-2.7.14", "2.8.1-2.8.29", "2.9.1-2.9.22", "2.10.1-2.10.30",
    "2.11.1-2.11.25", "2.12.1-2.12.25", "2.13.1-2.13.27", "2.14.1-2.14.29", "2.15.1-2.15.20",
    "2.16.1-2.16.47", "2.17.1-2.17.40", "2.18.1-2.18.32", "2.19.1-2.19.34", "2.20.1-2.20.37",
    "2.21.1-2.21.22", "2.22.1-2.22.37", "2.23.1-2.23.47", "2.24.1-2.24.29", "2.25.1-2.25.3",
    "2.26.1-2.26.7", "3.1.1-3.1.45", "3.2.1-3.2.18", "3.3.1-3.3.22", "3.4.1-3.4.12", "3.5.1-3.5.19",
    "3.6.1-3.6.10", "3.7.1-3.7.36", "3.8.1-3.8.7", "3.9.1-3.9.22", "3.10.1-3.10.20",
    "3.11.1-3.11.6", "3.12.1-3.12.17", "3.13.1-3.13.23", "3.14.1-3.14.14", "3.15.1-3.15.14",
    "3.16.1-3.16.16", "3.17.1-3.17.9", "3.18.1-3.18.9", "3.19.1-3.19.6", "3.20.1-3.20.19",
    "3.21.1-3.21.24", "3.22.1-3.22.109", "3.23.1-3.23.38", "3.24.1-3.24.118", "3.25.1-3.25.10",
    "3.26.1-3.26.39", "4.1.1-4.1.177", "4.2.1-4.2.10", "4.3.1-4.3.12", "4.4.1-4.4.48",
    "4.5.1-4.5.37", "4.6.1-4.6.38", "4.7.1-4.7.41", "4.8.1-4.8.43", "4.9.1-4.9.18",
    "4.10.1-4.10.36", "4.11.1-4.11.36", "4.12.1-4.12.21", "4.13.1-4.13.24"
]

philodemus_death_segments = [
    "1.1", "1.2", "1.15", "2.1", "2.3", "2.20", "2.21", "3.1", "3.4", "3.5", "3.29", "3.32",
    "4.1", "4.2", "4.6", "4.7", "4.30", "4.37", "5.1", "5.4", "5.6", "5.9",
    "6.1", "6.2", "6.4", "6.10", "6.15",  # original 6s
    "6.1", "6.11",  # added for 6 bis.1 and 6 bis.11
    "7.1", "7.2", "7.4", "7.6", "7.12", "7.15", "7.17", "7.28", "7.32", "7.38",
    "8.1", "8.6", "8.10", "8.13", "8.20", "8.26", "8.28", "8.30", "8.37",
    "9.1", "9.5", "9.8", "9.12", "9.14", "9.20",
    "10.1", "10.12", "10.30", "10.31", "10.36",  # original 10s
    "10.1", "10.3",  # added for 10 bis.1 and 10 bis.3
    "11.1", "11.2", "11.8", "11.10", "11.14", "11.20", "11.23", "11.26",
    "12.1", "12.2", "12.7", "12.11", "12.15", "12.31", "12.34",
    "13.1", "13.3", "13.9", "13.10", "13.13", "13.28", "13.33", "13.36",
    "14.1", "14.2", "14.5", "14.10", "14.28", "14.35", "14.38",
    "15.1", "15.5", "15.10", "15.24", "15.30", "15.35",
    "16.1", "16.2", "16.5", "16.7", "16.10", "16.15", "16.23", "16.37", "16.38", "16.39",
    "17.1", "17.3", "17.11", "17.13", "17.16", "17.27", "17.30", "17.32",
    "18.1", "18.5", "18.9", "18.15", "18.24", "18.30", "18.33",
    "19.1", "19.3", "19.6", "19.11", "19.15", "19.27", "19.33",
    "20.1", "20.5", "20.10", "20.11", "20.14", "20.22", "20.26", "20.27",
    "21.1", "21.5", "21.6", "21.12", "21.17", "21.26",
    "22.1", "22.5", "22.9", "22.12", "22.29", "22.37",
    "23.1", "23.2", "23.8", "23.19", "23.25", "23.29", "23.35", "23.36",
    "24.1", "24.5", "24.10", "24.17", "24.22", "24.25", "24.28", "24.31",
    "25.1", "25.2", "25.10", "25.12", "25.18", "25.20", "25.27", "25.30", "25.34", "25.37",
    "26.1", "26.3", "26.7", "26.10", "26.14", "26.19", "26.28", "26.30", "26.37",
    "27.1", "27.8", "27.15", "27.29", "27.33", "27.35",
    "28.1", "28.5", "28.14", "28.20", "28.27", "28.30", "28.32",
    "29.1", "29.2", "29.15", "29.22", "29.26", "29.27", "29.30",
    "30.1", "30.5", "30.7", "30.24", "30.27",
    "31.1", "31.5", "31.11", "31.20", "31.28", "31.30",
    "32.1", "32.2", "32.9", "32.16", "32.20", "32.24", "32.28", "32.31", "32.36",
    "33.1", "33.5", "33.9", "33.10", "33.23", "33.25", "33.30", "33.31", "33.37",
    "34.1", "34.4", "34.9", "34.15", "34.21", "34.29", "34.35", "34.38",
    "35.1", "35.6", "35.11", "35.19", "35.25", "35.30", "35.34", "35.39",
    "36.1", "36.8", "36.12", "36.17", "36.23", "36.25", "36.27", "36.31", "36.37", "36.40",
    "37.1", "37.10", "37.12", "37.15", "37.18", "37.27", "37.39",
    "38.1", "38.3", "38.5", "38.10", "38.14", "38.25", "38.30",
    "39.1", "39.6", "39.15"
]

# full structure for the damascus document
damascus_full_structure = (
    [f"{i}.{j}" for i in range(1, 9) for j in range(1, 22)] +  # cols I–VIII (1–8)
    [f"{i}.{j}" for i in range(15, 17) for j in range(1, 21)] +  # cols XV–XVI (15–16)
    [f"{i}.{j}" for i in range(9, 15) for j in range(1, 24)] +  # cols IX–XIV (9–14)
    [f"19.{j}" for j in range(1, 36)] +  # col XIX
    [f"20.{j}" for j in range(1, 35)]  # col XX
)

seneca_full_structure = expand_ranges(seneca_ranges)
epictetus_full_structure = expand_ranges(epictetus_ranges)
aristides_full_structure = [str(i) for i in range(1, 48)]
philodemus_piety_full_structure = [str(i) for i in range(1, 87)]

# define full section lists for structure-based deciles
structure_maps = {
    "cleaned_seneca_natural_questions.csv": seneca_full_structure,
    "cleaned_epictetus_discourses.csv": epictetus_full_structure,
    "cleaned_aelius_aristides_panathenaicus.csv": aristides_full_structure,
    "cleaned_philodemus_on_piety.csv": philodemus_piety_full_structure,
    "cleaned_damascus_document.csv": damascus_full_structure,
    "cleaned_philodemus_on_death.csv": philodemus_death_segments
}

for f in non_biblical_files:
    path = os.path.join(clean_data_dir, f)
    text_name = f.replace("cleaned_", "").replace(".csv", "").replace("_", " ").title()
    out = os.path.join(output_dir, f.replace(".csv", "_with_deciles.csv"))

    if f in structure_maps:
        assign_row_deciles_structural(path, out, text_name, structure_maps[f])
    else:
        assign_row_deciles(path, out, text_name)

# merge biblical feature files with verse-based deciles
biblical_files = {
    "Romans": "cleaned_romans.csv",
    "Galatians": "cleaned_galatians.csv",
    "1 Thessalonians": "cleaned_1thessalonians.csv",
    "2 Thessalonians": "cleaned_2thessalonians.csv",
    "1 Corinthians": "cleaned_1corinthians.csv",
    "2 Corinthians": "cleaned_2corinthians.csv",
    "Philippians": "cleaned_philippians.csv",
    "Philemon": "cleaned_philemon.csv",
    "Hebrews": "cleaned_hebrews.csv"
}

for letter, filename in biblical_files.items():
    fpath = os.path.join(clean_data_dir, filename)
    explode_and_merge_biblical(fpath, biblical_deciles, letter)
