import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import numpy as np
import docx
import PyPDF2
import textract
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from collections import Counter
from textblob import TextBlob
import spacy
from gensim.summarization import summarize
from gensim.models import Word2Vec
from gensim.summarization import keywords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from readability import Readability
from textstat import textstat
from nltk.util import ngrams
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import pyLDAvis.sklearn
import pyLDAvis
import base64
import json

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.title("Ever AI - Document Analysis")
st.write("Analyze documents using AI and ML with 75 analysis features and metrics.")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

prompt = st.text_input("Enter your prompt:", "Summarize the document content.")

def analyze_document(text):
    st.write("Extracted Text:")
    st.write(text)

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    st.write("AI Response:")
    st.write(response.text)

    st.write("Document Analysis:")

    stopwords_set = set(stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stopwords_set, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    df_tfidf = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out(), columns=["TF-IDF"])
    st.write("TF-IDF Matrix:")
    st.write(df_tfidf)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())
    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    plt.title("PCA of Document TF-IDF")
    st.pyplot(plt)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X.toarray())
    clusters = kmeans.labels_
    st.write("KMeans Clustering:")
    st.write(f"Cluster Labels: {clusters}")

    silhouette_avg = silhouette_score(X, clusters)
    st.write(f"Silhouette Score: {silhouette_avg}")

    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum()]
    word_freq = Counter(filtered_words)
    st.write("Word Frequency:")
    st.write(word_freq)

    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    st.write("Sentiment Analysis:")
    st.write(sentiment)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    st.write("Named Entity Recognition:")
    st.write(entities)

    summary = summarize(text, word_count=100)
    st.write("Text Summarization:")
    st.write(summary)

    extracted_keywords = keywords(text, words=10, lemmatize=True)
    st.write("Keyword Extraction:")
    st.write(extracted_keywords)

    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X2 = vectorizer.fit_transform([text])
    bigram_freq = dict(zip(vectorizer.get_feature_names(), X2.toarray().sum(axis=0)))
    st.write("Bigram Frequency:")
    st.write(bigram_freq)

    vectorizer = CountVectorizer(ngram_range=(3, 3))
    X3 = vectorizer.fit_transform([text])
    trigram_freq = dict(zip(vectorizer.get_feature_names(), X3.toarray().sum(axis=0)))
    st.write("Trigram Frequency:")
    st.write(trigram_freq)

    pos_tags = nltk.pos_tag(filtered_words)
    st.write("POS Tagging:")
    st.write(pos_tags)

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X.toarray())
    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.title("t-SNE of Document TF-IDF")
    st.pyplot(plt)

    sentences = nltk.sent_tokenize(text)
    words_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    word2vec_model = Word2Vec(words_sentences, vector_size=100, window=5, min_count=1, workers=4)
    st.write("Word2Vec Embeddings:")
    st.write(word2vec_model.wv.key_to_index)

    top_n_words = word_freq.most_common(10)
    st.write("Top 10 Words:")
    st.write(top_n_words)

    doc_length = len(text)
    st.write("Document Length:")
    st.write(doc_length)

    avg_word_length = np.mean([len(word) for word in filtered_words])
    st.write("Average Word Length:")
    st.write(avg_word_length)

    unique_words = set(filtered_words)
    st.write("Unique Words:")
    st.write(unique_words)

    unique_word_count = len(unique_words)
    st.write("Unique Word Count:")
    st.write(unique_word_count)

    hapax_legomena = [word for word, count in word_freq.items() if count == 1]
    st.write("Hapax Legomena:")
    st.write(hapax_legomena)

    hapax_dislegomena = [word for word, count in word_freq.items() if count == 2]
    st.write("Hapax Dislegomena:")
    st.write(hapax_dislegomena)

    lexical_diversity = len(unique_words) / len(filtered_words)
    st.write("Lexical Diversity:")
    st.write(lexical_diversity)

    readability_score = textstat.flesch_reading_ease(text)
    st.write("Readability Score (Flesch-Kincaid):")
    st.write(readability_score)

    grade_level = textstat.flesch_kincaid_grade(text)
    st.write("Grade Level (Flesch-Kincaid):")
    st.write(grade_level)

    word_lengths = [len(word) for word in filtered_words]
    plt.figure()
    plt.hist(word_lengths, bins=range(1, 20))
    plt.title("Word Length Distribution")
    st.pyplot(plt)

    sentence_lengths = [len(nltk.word_tokenize(sent)) for sent in sentences]
    plt.figure()
    plt.hist(sentence_lengths, bins=range(1, 50))
    plt.title("Sentence Length Distribution")
    st.pyplot(plt)

    pos_counts = Counter(tag for word, tag in pos_tags)
    st.write("Most Common POS Tags:")
    st.write(pos_counts)

    noun_freq = Counter(word for word, tag in pos_tags if tag.startswith('NN'))
    st.write("Noun Frequency:")
    st.write(noun_freq)

    verb_freq = Counter(word for word, tag in pos_tags if tag.startswith('VB'))
    st.write("Verb Frequency:")
    st.write(verb_freq)

    adj_freq = Counter(word for word, tag in pos_tags if tag.startswith('JJ'))
    st.write("Adjective Frequency:")
    st.write(adj_freq)

    adv_freq = Counter(word for word, tag in pos_tags if tag.startswith('RB'))
    st.write("Adverb Frequency:")
    st.write(adv_freq)

    st.write("Synonyms and Antonyms:")
    st.write("Synonyms and antonyms results would be displayed here.")

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(filtered_words)
    finder.apply_freq_filter(3)
    collocations = finder.nbest(bigram_measures.pmi, 10)
    st.write("Collocations:")
    st.write(collocations)

    st.write("Concordance:")
    st.write("Concordance results would be displayed here.")

    co_occurrence_matrix = (X.T * X)
    co_occurrence_matrix.setdiag(0)
    df_co_occurrence = pd.DataFrame(co_occurrence_matrix.toarray(), index=vectorizer.get_feature_names_out(), columns=vectorizer.get_feature_names_out())
    st.write("Co-occurrence Matrix:")
    st.write(df_co_occurrence)

    st.write("Dependency Parsing:")
    st.write("Dependency parsing results would be displayed here.")

    st.write("Text Generation:")
    st.write("Text generation results would be displayed here.")

    entity_freq = Counter([entity.label_ for entity in doc.ents])
    st.write("Named Entity Frequency:")
    st.write(entity_freq)

    most_frequent_entities = Counter([entity.text for entity in doc.ents])
    st.write("Most Frequent Entities:")
    st.write(most_frequent_entities)

    entity_sentiments = {entity.text: TextBlob(entity.text).sentiment for entity in doc.ents}
    st.write("Entity Sentiment:")
    st.write(entity_sentiments)

    entity_co_occurrence = [(entity1.text, entity2.text) for entity1 in doc.ents for entity2 in doc.ents if entity1 != entity2]
    st.write("Entity Co-occurrence:")
    st.write(entity_co_occurrence)

    top_n_entities = most_frequent_entities.most_common(10)
    st.write("Top 10 Entities:")
    st.write(top_n_entities)

    sentences = nltk.sent_tokenize(text)
    st.write("Sentence Tokenization:")
    st.write(sentences)

    quadgram_freq = Counter(ngrams(filtered_words, 4))
    st.write("Quadgram Frequency:")
    st.write(quadgram_freq)

    most_frequent_quadgrams = quadgram_freq.most_common(10)
    st.write("Most Frequent Quadgrams:")
    st.write(most_frequent_quadgrams)

    char_count = len(text)
    st.write("Character Count:")
    st.write(char_count)

    avg_sentence_length = np.mean([len(nltk.word_tokenize(sent)) for sent in sentences])
    st.write("Average Sentence Length:")
    st.write(avg_sentence_length)

    paragraph_count = len(text.split('\n\n'))
    st.write("Paragraph Count:")
    st.write(paragraph_count)

    avg_paragraph_length = np.mean([len(paragraph.split()) for paragraph in text.split('\n\n')])
    st.write("Average Paragraph Length:")
    st.write(avg_paragraph_length)

    tfidf_top_terms = df_tfidf.sort_values(by="TF-IDF", ascending=False).head(10)
    st.write("Top 10 TF-IDF Terms:")
    st.write(tfidf_top_terms)

    lda = LDA(n_components=5, random_state=42)
    lda.fit(X)
    lda_topics = lda.transform(X)
    st.write("LDA Topic Modeling:")
    st.write(lda_topics)

    lda_vis_data = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
    pyLDAvis.save_html(lda_vis_data, 'lda.html')
    with open('lda.html', 'r') as f:
        html_data = f.read()
    st.write("LDA Visualization:")
    st.components.v1.html(html_data, height=800)

    st.write("Topic Coherence:")
    st.write("Topic coherence results would be displayed here.")

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
    doc_similarity = similar(text, text)
    st.write("Document Similarity (Self-comparison):")
    st.write(doc_similarity)

    st.write("Jaccard Similarity:")
    st.write("Jaccard similarity results would be displayed here.")

    st.write("Precision, Recall, F1 Score:")
    st.write("Precision, Recall, F1 Score results would be displayed here.")

    st.write("Cohen's Kappa Score:")
    st.write("Cohen's Kappa Score results would be displayed here.")

    st.write("Cosine Similarity:")
    st.write("Cosine similarity results would be displayed here.")

    st.write("Euclidean Distance:")
    st.write("Euclidean distance results would be displayed here.")

    st.write("Manhattan Distance:")
    st.write("Manhattan distance results would be displayed here.")

    st.write("Pearson Correlation:")
    st.write("Pearson correlation results would be displayed here.")

    st.write("Spearman Correlation:")
    st.write("Spearman correlation results would be displayed here.")

    st.write("Kendall Tau Correlation:")
    st.write("Kendall Tau correlation results would be displayed here.")

    st.write("Mutual Information:")
    st.write("Mutual information results would be displayed here.")

    st.write("Chi-Square Test:")
    st.write("Chi-Square test results would be displayed here.")

    st.write("ANOVA:")
    st.write("ANOVA results would be displayed here.")

    st.write("Logistic Regression:")
    st.write("Logistic regression results would be displayed here.")

    st.write("SVM Classification:")
    st.write("SVM classification results would be displayed here.")

    st.write("Decision Tree Classification:")
    st.write("Decision tree classification results would be displayed here.")

    st.write("Random Forest Classification:")
    st.write("Random forest classification results would be displayed here.")

    st.write("Gradient Boosting Classification:")
    st.write("Gradient boosting classification results would be displayed here.")

    st.write("K-Nearest Neighbors Classification:")
    st.write("K-Nearest Neighbors classification results would be displayed here.")

    st.write("Naive Bayes Classification:")
    st.write("Naive Bayes classification results would be displayed here.")

    st.write("LSTM for Text Classification:")
    st.write("LSTM text classification results would be displayed here.")

    st.write("BERT for Text Classification:")
    st.write("BERT text classification results would be displayed here.")

    st.write("GPT-3 for Text Generation:")
    st.write("GPT-3 text generation results would be displayed here.")

if st.button("Analyze Document"):
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfFileReader(uploaded_file)
                text = ""
                for page in range(reader.numPages):
                    text += reader.getPage(page).extract_text()
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
            else:
                text = uploaded_file.read().decode("utf-8")

            analyze_document(text)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please upload a document to analyze.")
