from flask import Flask, request, render_template, send_file, redirect, url_for
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from google_play_scraper import reviews, Sort
import nltk
import os
import datetime
import tempfile
import uuid
from urllib.parse import urlparse, parse_qs

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load pre-trained models
classifier = joblib.load('saved_models/svm_model_rbf.pkl')
vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')



def extract_app_id(input_text):
    if input_text.startswith("http"):
        parsed_url = urlparse(input_text)
        query_params = parse_qs(parsed_url.query)
        input_text = query_params.get('id', [None])[0]
    else:
        input_text = input_text

    return input_text

def preprocess_text(text):
    """Preprocess text by cleaning, tokenizing, and lemmatizing."""
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_reviews(reviews_data):
    """Preprocess reviews data."""
    reviews_list = [
        {
            'userName': review['userName'],
            'content': review['content'],
            'score': review['score'],
            'thumbsUpCount': review['thumbsUpCount'],
            'reviewCreatedVersion': review.get('reviewCreatedVersion', 'N/A')
        }
        for review in reviews_data
    ]
    df = pd.DataFrame(reviews_list)
    df['cleaned_content'] = df['content'].apply(preprocess_text)
    return df

def visualize_sentiment_distribution(df):
    """Visualize the distribution of sentiments."""
    sentiment_counts = df['predicted_sentiment'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'blue'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_wordcloud(df):
    """Generate and display a word cloud from the cleaned reviews."""
    text = ' '.join(df['cleaned_content'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    img.seek(0)
    wordcloud_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return wordcloud_url

def visualize_sentiment_by_version(df):
    """Visualize average sentiment score by version."""
    version_sentiment = df.groupby('reviewCreatedVersion')['predicted_sentiment'].mean().sort_index()
    plt.figure(figsize=(10, 5))
    plt.plot(version_sentiment.index, version_sentiment.values, marker='o', linestyle='-', color='b')
    plt.title('Average Sentiment Score by Version')
    plt.xlabel('App Version')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=90)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    log_access(request)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    log_access(request)
    input_text = request.form['app_id']
    # Generate a unique temporary directory for this session
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(tempfile.gettempdir(), session_id)
    os.makedirs(session_dir)
    app_id = extract_app_id(input_text)
    if not app_id:
        return redirect('/')
    reviews_data = scrape_reviews(app_id, num_reviews=300)
    if reviews_data:
        df = preprocess_reviews(reviews_data)
        X = vectorizer.transform(df['cleaned_content']).toarray()
        df['predicted_sentiment'] = classifier.predict(X)
        sentiment_plot = visualize_sentiment_distribution(df)
        wordcloud_plot = generate_wordcloud(df)
        sentiment_by_version_plot = visualize_sentiment_by_version(df)
        df.to_csv(os.path.join(session_dir, 'results.csv'), index=False, encoding='utf-8')
        return render_template('result.html', sentiment_plot=sentiment_plot, wordcloud_plot=wordcloud_plot, sentiment_by_version_plot=sentiment_by_version_plot, app_id=app_id, session_id=session_id, tables=[df.head(10).to_html(classes='data')], titles=df.columns.values)
    else:
        return redirect('/')

@app.route('/download/<session_id>')
def download(session_id):
    session_dir = os.path.join(tempfile.gettempdir(), session_id)
    file_path = os.path.join(session_dir, 'results.csv')
    return send_file(file_path, as_attachment=True)

def scrape_reviews(app_id, lang='id', country='id', num_reviews=300):
    """Scrape reviews from Google Play Store."""
    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=num_reviews
        )
        print(f"Scraping {num_reviews} reviews for app_id={app_id} succeeded.")
        return result
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        return []

def log_access(request):
    """
    Logs the user's IP address, timestamp, and HTTP method to a log file.
    """
    log_file = "access_log.txt"
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)
    ip_address = request.remote_addr
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    http_method = request.method
    input_data = request.form.to_dict()
    user_agent = str(request.user_agent)
    with open(log_path, "a") as log_file:
        log_entry = f"{timestamp} | {ip_address} | {http_method} | {input_data} | {user_agent}"
        log_file.write(log_entry + "\n")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=15011)
