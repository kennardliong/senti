A sentiment analyzer app. 


<img width="981" alt="Screenshot 2025-02-09 at 17 21 35" src="https://github.com/user-attachments/assets/3ef9a1e7-0f26-48a9-abc4-a04fc3772cd3" />

How Our Sentiment Analysis Works
Our sentiment analysis tool combines information to get the most accurate and up-to-date analysis.

Gathering Reddit Data: We collect 50 recent posts from Reddit related to your search term to gauge public opinion.
Analyzing Reddit Post Titles: We analyse the titles of recent posts, using TF-IDF vectorizer (to account for relevance), stop word removal (filtering meaningless words), and exclude neutral scores (to reduce noise) to refine the analysis.
LLM-Based Analysis (Gemini): We also use Gemini to consider historical significance, recent events, and public perception from both social and corporate standpoints.
Combining Scores: The final sentiment score is a weighted average of the Reddit-based model's score and the LLM-based score. By default, we give each component equal weight (50/50).
Examples
Coca-Cola: Score ≈ 55. Despite its popularity, there are considerations to factor in like health concerns from its sugary composition.
NVIDIA: Score ≈ 62. A major player in graphics and AI, generally viewed favorably, but subject to market volatility and supply chain concerns.
Uncertainty and Limitations
Sentiment analysis is inherently uncertain. Public sentiment changes over time due to news events, social trends, and other factors. Our tool provides a snapshot of sentiment based on recent data.

Additionally, our model is not perfect. Sentiment is nuanced. We do our best to provide results that reflect the real world, but it isn't always certain.

Get scikit-learn, google.generativeai, dotenv, langdetect, nltk.sentiment.vader, corpus/stopwords, flask, and praw API (reddit) to run!
