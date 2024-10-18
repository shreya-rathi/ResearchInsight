import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Step 1: Sample data related to "XR in Education"
df = pd.read_csv('scopus2.csv')
df['Abstract'] = df['Abstract'].fillna('')
df['Author Keywords'] = df['Author Keywords'].fillna('')
documents = (df['Abstract'] + ' ' + df['Author Keywords']).tolist()

# Step 2: Create a Document-Term Matrix
vectorizer = CountVectorizer(stop_words='english')
dtm = vectorizer.fit_transform(documents)

# Step 3: Train the LDA Model
num_topics = 1  # We are interested in one overarching topic
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(dtm)

# Step 4: Display the topics and words associated with the topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10  # Number of words to display for the topic
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

# Step 5: Generate a Word Cloud
topic_words = lda.components_[0]  # Get the first (and only) topic's word distribution
words = vectorizer.get_feature_names_out()
word_dict = {words[i]: topic_words[i] for i in range(len(words))}

# Specify a path to a TrueType font (update the font path as needed)
font_path = "../Arial.ttf"  

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate_from_frequencies(word_dict)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
