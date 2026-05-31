import pandas as pd
import os
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import numpy as np

# get topic distributions per decile
def dense_vector(dist, num_topics):
    vec = [0.0] * num_topics
    for topic_id, prob in dist:
        vec[topic_id] = prob
    return vec


if __name__ == '__main__':
    # set data folder
    input_dir = "deciles"

    # list of paul files
    paul_files = [
        "cleaned_1 corinthians_with_deciles.csv",
        "cleaned_2 corinthians_with_deciles.csv",
        "cleaned_1 thessalonians_with_deciles.csv",
        "cleaned_galatians_with_deciles.csv",
        "cleaned_philemon_with_deciles.csv",
        "cleaned_philippians_with_deciles.csv",
        "cleaned_romans_with_deciles.csv",
    ]

    # gather rhetorical features by decile
    documents = []
    doc_labels = []

    for filename in paul_files:
        path = os.path.join(input_dir, filename)
        df = pd.read_csv(path)

        grouped = defaultdict(list)
        for _, row in df.iterrows():
            if pd.isna(row["Characteristic"]) or pd.isna(row["Decile"]):
                continue
            decile = int(row["Decile"])
            feature = row["Characteristic"].strip()
            grouped[decile].append(feature)

        for decile, features in sorted(grouped.items()):
            documents.append(features)
            label = filename.replace("_with_deciles.csv", "").replace("cleaned_", "")
            doc_labels.append(f"{label} - Decile {decile}")

    # convert to gensim format
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # train models with hardcoded topic numbers and evaluate coherence
    topic_numbers = [2, 5, 10]
    models = {}
    coherences = []

    for k in topic_numbers:
        print(f"\nTraining model with {k} topics:")
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=10,
            random_state=42
        )
        models[k] = model

        # calculate coherence
        coherence = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v').get_coherence()
        coherences.append((k, coherence))

        print(f"Model with {k} topics - Coherence: {coherence:.4f}")
        print(f"Topics for {k}-topic model:")
        topics = model.print_topics(num_words=6)
        for i, topic in topics:
            print(f"  Topic {i}: {topic}")

    print("\n" + "=" * 50)
    print("COHERENCE COMPARISON:")
    for k, coherence in coherences:
        print(f"{k} topics: {coherence:.4f}")

    # manually select which model to use for further analysis
    selected_k = 5  # change this to whichever number of topics you want
    if selected_k in models:
        lda_model = models[selected_k]

        print(f"\nUsing {selected_k}-topic model for analysis:")
        topics = lda_model.print_topics(num_words=6)
        for i, topic in topics:
            print(f"Topic {i}: {topic}")

        # calculate topic distributions and baseline
        topic_dists = [lda_model.get_document_topics(doc, minimum_probability=0.0) for doc in corpus]
        dense_dists = [dense_vector(dist, lda_model.num_topics) for dist in topic_dists]
        paul_baseline = np.mean(dense_dists, axis=0)

        print(f"\nPaul Baseline Vector:")
        print(paul_baseline)

        ### COMPARANDA ###

        # now compare other texts to paul's baseline
        comparanda_files = [
            "cleaned_epictetus_discourses_with_deciles.csv",
            "cleaned_seneca_natural_questions_with_deciles.csv",
            "cleaned_philodemus_on_piety_with_deciles.csv",
            "cleaned_philodemus_on_death_with_deciles.csv",
            "cleaned_aelius_aristides_panathenaicus_with_deciles.csv",
            "cleaned_damascus_document_with_deciles.csv",
            "cleaned_4maccabees_with_deciles.csv",
            "cleaned_2 thessalonians_with_deciles.csv",
            "cleaned_hebrews_with_deciles.csv"
        ]

        for filename in comparanda_files:
            path = os.path.join("deciles", filename)
            df = pd.read_csv(path)

            # make sure data is clean
            df = df.drop_duplicates(subset=["Segment", "Characteristic"])
            if "Decile" not in df.columns:
                raise ValueError(f"{filename} is missing Decile info")

            # build documents by decile
            grouped = defaultdict(list)
            for _, row in df.iterrows():
                if pd.isna(row["Characteristic"]) or pd.isna(row["Decile"]):
                    continue
                decile = int(row["Decile"])
                feature = row["Characteristic"].strip()
                grouped[decile].append(feature)

            # convert to bow using the same dictionary from Paul
            comp_docs = []
            comp_labels = []
            for decile, features in sorted(grouped.items()):
                comp_docs.append(dictionary.doc2bow(features))
                label = filename.replace(".csv", "").replace("cleaned_", "")
                comp_labels.append(f"{label} - Decile {decile}")

            # get topic distributions from paul-trained model
            comp_dists = [lda_model.get_document_topics(doc, minimum_probability=0.0) for doc in comp_docs]
            dense_comp_dists = [dense_vector(dist, lda_model.num_topics) for dist in comp_dists]

            # get average topic distribution for this text
            avg_dist = np.mean(dense_comp_dists, axis=0)

            # print most prominent topics in this text
            print(f"\nAverage topic distribution for {filename}:")
            sorted_topics = sorted(enumerate(avg_dist), key=lambda x: x[1], reverse=True)

            for topic_id, weight in sorted_topics:
                terms = lda_model.show_topic(topic_id, topn=6)
                term_str = " + ".join([f"{w:.3f}*\"{t}\"" for t, w in terms])
                print(f"Topic {topic_id} ({weight:.3f}): {term_str}")

            # ex: in 4maccabees, "Topic 0 (0.652)" means that, on average, Topic 0 makes up 65.2% of the topic
            # distribution across the deciles of 4maccabees

            # compare with paul baseline using js divergence
            def jensen_shannon(p, q):
                p = np.array(p)
                q = np.array(q)
                m = 0.5 * (p + q)
                return 0.5 * (np.sum(p * np.log2(p / m + 1e-12)) + np.sum(q * np.log2(q / m + 1e-12)))


            print(f"\nJensen-Shannon Divergence vs. Paul Baseline for {filename}:")
            for label, vec in zip(comp_labels, dense_comp_dists):
                jsd = jensen_shannon(vec, paul_baseline)
                print(f"{label}: {jsd:.4f}")
