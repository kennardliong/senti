import praw
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob  # We'll still keep TextBlob for language detection (see below)
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import VADER
from decimal import Decimal, ROUND_HALF_UP
from langdetect import detect #added language detection
from nltk.corpus import stopwords #stop word removal
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai # Import the Gemini library
import os #for getting the api_key
from dotenv import load_dotenv # Import load_dotenv

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

# Reddit API credentials (replace with your actual credentials)
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT")  # Required
reddit_username = os.getenv("REDDIT_USERNAME")  # Needed to read the token, might be able to get around this
reddit_password = os.getenv("REDDIT_PASSWORD")

reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent,
    username=reddit_username,
    password=reddit_password,
)
user = reddit.user.me()
print(f"Successfully connected to Reddit as: {user.name}")

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Download stopwords only if you haven't already
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def get_reddit_posts(query, num_posts=100):
    """Fetches Reddit posts related to a query."""
    posts = []
    try:
        for submission in reddit.subreddit("all").search(query, limit=num_posts): # Search all subreddits
            # Use the submission's title *and* selftext for sentiment analysis
            post_text = submission.title # only gets the title now  # Combine title and body
            try:
                if detect(post_text) == 'en':  # Check if post is in English
                    posts.append(post_text)
                else:
                    print(f"Skipping non-English post: {post_text[:50]}...")
            except:
                print(f"Error detecting language for post: {post_text[:50]}...")


    except Exception as e:
        print(f"Error fetching posts: {e}")
        return []
    return posts

def remove_stopwords(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return ". ".join(filtered_words)


# Initialize Gemini
# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) #need to create the env key in the code
model = genai.GenerativeModel("gemini-1.5-flash")

def get_llm_sentiment_score(term):
    prompt = f"""Given the word '{term}', analyze its sentiment by considering historical significance, recent events, and public perception, both from a more social and corporate standpoint. Focus on providing a neutral perspective, with attention to any mixed opinions or controversies, without leaning excessively positive or negative. Respond in the following format:

score: [sentiment score from 0 to 100, with one decimal place]
justification: [factual reasoning for the score, aiming for balance, a one-sentence response, and wit]

Example:
score: 62.5
justification: A popular brand with some bad PR from its business practices.

Now, analyze '{term}':"""

    try:
        response = model.generate_content(prompt)
        llm_output = response.text
        print(f"LLM Output: {llm_output}")

        # Extract sentiment score from LLM output and the description
        match = re.search(r"score:\s*(\d+\.\d+)\s*justification:\s*(.*)", llm_output, re.DOTALL)

        if match:
            llm_score = float(match.group(1))
            llm_justification = match.group(2).strip() # Get the justification
            print("AAAA"+llm_justification)
            return llm_score, llm_justification
        else:
            print("Could not extract sentiment score and justification from LLM output.")
            return 50, "Could not extract sentiment score and justification from LLM output."  # Default LLM score if extraction fails, and tell llm output that
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        return 50, f"Error during LLM analysis: {e}" #tell llm why it failed

@app.route('/')
def index():
    return render_template('index.html')

def combine_scores(model_score, llm_score, user_score=None):
    w_ms = 0.5
    w_ls = 0.5
    w_us = 0.0  # Default: No user score

    if user_score is not None:
        w_us = 0.25
        w_ls = 0.25
        w_ms = 0.5
    else:
        user_score = 0 #assign it to 0 if it's nothing

    total_weight = w_ms + w_ls + w_us
    combined_score = (w_ms * model_score + w_ls * llm_score + w_us * user_score) / total_weight

    return combined_score

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.form['text']
        print(f"Received text: {text}")

        posts = get_reddit_posts(text)
        print(f"get_reddit_posts returned: {posts}")

        if not posts:
            print("No posts found, returning default score")
            #now llm, and then combine
            llm_score, llm_output = get_llm_sentiment_score(text)
            combined_score = combine_scores(50, llm_score)
            rounded_score = round(combined_score, 2)
            return jsonify({'score': float(rounded_score), 'llm_output': llm_output})

        # 1. Prepare the Data and TF-IDF Vectorizer
        cleaned_posts = [remove_stopwords(post) for post in posts] # Applying stop word removal
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_posts) # Fit and Transform - make sure that these posts are clean!!!

        # 2. Get TF-IDF Scores for the Search Term
        tfidf_scores = []
        term_index = vectorizer.vocabulary_.get(text.lower())

        if term_index is not None:
            for i in range(len(posts)):
                tfidf_scores.append(tfidf_matrix[i, term_index])
        else:
            print("Search term not found in any documents.")
            tfidf_scores = [0] * len(posts)
        
        
        if tfidf_scores:
            # 3. Calculate Weighted Sentiment, Excluding Neutral Posts
            total_weighted_compound_score = 0
            total_tfidf_score = 0 # added this
            num_meaningful_posts = 0 # Keep track of the number of non-neutral posts

            for i, post in enumerate(posts):
                vs = sid.polarity_scores(post)
                compound_score = vs['compound']
                tfidf_score = tfidf_scores[i]
                weighted_compound_score = compound_score * tfidf_score  # Calculate weighted score
                sentiment_score_for_post = (compound_score + 1) * 50 #raw score for just the post

                # Check if the sentiment score is *exactly* 50
                if abs(sentiment_score_for_post - 50) > 1e-6: # Use a small tolerance for floating-point comparison
                    total_weighted_compound_score += weighted_compound_score
                    total_tfidf_score += tfidf_score #added this for normalization purposes
                    num_meaningful_posts += 1
                    print(f"Post: {post[:50]}... Compound Score: {compound_score}, TF-IDF Score: {tfidf_score}, Sentiment Score: {sentiment_score_for_post}")
                else:
                    print(f"Skipping neutral post: {post[:50]}... Sentiment Score: {sentiment_score_for_post}")


            # 4. Calculate Average Weighted Sentiment (Handling the case where all posts are neutral)
            if num_meaningful_posts > 0:
                average_weighted_compound_score = total_weighted_compound_score / total_tfidf_score if total_tfidf_score > 0 else 0
            else:
                average_weighted_compound_score = 0
                print("All posts were neutral; returning default score.")



            model_score = (average_weighted_compound_score + 1) * 50 #assign the model
        else:
            model_score = 50

        llm_score, llm_output = get_llm_sentiment_score(text)

        combined_score = combine_scores(model_score, llm_score)

        rounded_score = round(combined_score,2)
        return jsonify({'score': float(rounded_score), 'llm_output': llm_output})  # Return llm output as well

    except Exception as e:
        print(f"Error within /analyze route: {e}")
        return jsonify({'score': 50, 'error': str(e)})
    
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)