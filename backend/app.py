from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Sample test data for different subjects
SAMPLE_QUESTIONS = {
    "python": {
        "question": "Explain the concept of inheritance in Python with an example.",
        "sample_answers": [
            """Inheritance is a mechanism that allows a class to inherit attributes and methods from another class.
            For example, class Dog(Animal) inherits from Animal class, getting its properties like 'speak' and 'eat'.""",
            """Python inheritance is when one class takes on the attributes and methods of another class.
            The class that inherits is called child class, while the class being inherited from is the parent class."""
        ]
    },
    "database": {
        "question": "What is normalization in database design?",
        "sample_answers": [
            """Database normalization is the process of organizing data to minimize redundancy.
            It involves dividing large tables into smaller ones and defining relationships between them.""",
            """Normalization is a technique used in database design to reduce data redundancy and ensure data integrity.
            It follows normal forms like 1NF, 2NF, and 3NF to structure the database efficiently."""
        ]
    },
    "networking": {
        "question": "Explain TCP/IP protocol.",
        "sample_answers": [
            """TCP/IP is a suite of communication protocols used to interconnect network devices on the internet.
            TCP ensures reliable data delivery while IP handles addressing and routing.""",
            """The TCP/IP protocol is the fundamental communication protocol of the internet.
            It uses a layered approach and includes protocols like TCP for data transfer and IP for routing."""
        ]
    }
}

# File to store submitted answers
SUBMISSIONS_FILE = 'submissions.json'

def load_submissions():
    """Load previous submissions from file."""
    if os.path.exists(SUBMISSIONS_FILE):
        with open(SUBMISSIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_submission(question, answer, similarity_score):
    """Save a new submission to file."""
    submissions = load_submissions()
    submissions.append({
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'similarity_score': similarity_score
    })
    with open(SUBMISSIONS_FILE, 'w') as f:
        json.dump(submissions, f, indent=2)

def preprocess_text(text):
    """Preprocess the text by converting to lowercase and removing stopwords."""
    try:
        words = text.lower().split()
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in words if word not in stop_words])
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return text

def check_plagiarism(answer, question_type):
    """Check for plagiarism using cosine similarity against corpus and previous submissions."""
    try:
        # Get sample answers for the question type
        corpus = SAMPLE_QUESTIONS.get(question_type, {}).get('sample_answers', [])
       
        # Add previous submissions to the corpus
        submissions = load_submissions()
        previous_answers = [sub['answer'] for sub in submissions]
        all_texts = corpus + previous_answers
       
        if not all_texts:
            return 0.0, True

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([answer] + all_texts)
        similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
        max_similarity = float(max(similarity_scores) * 100) if len(similarity_scores) > 0 else 0.0
       
        return max_similarity, max_similarity < 30

    except Exception as e:
        print(f"Error in plagiarism check: {str(e)}")
        return 0.0, True

def calculate_english_quality(text):
    """Calculate English quality score based on vocabulary diversity and length."""
    try:
        words = text.lower().split()
        if len(words) < 20:  # Penalize very short answers
            return float(0.5)
        vocab_diversity = len(set(words)) / len(words)
        return float(vocab_diversity)
    except Exception as e:
        print(f"Error in English quality calculation: {str(e)}")
        return 0.0

@app.route('/questions', methods=['GET'])
def get_questions():
    """Return available sample questions."""
    return jsonify({
        'questions': [
            {'type': qtype, 'question': data['question']}
            for qtype, data in SAMPLE_QUESTIONS.items()
        ]
    })

@app.route('/evaluate', methods=['POST'])
def evaluate_answer():
    """Evaluate the submitted answer."""
    try:
        data = request.json
        if not data or 'answer' not in data or 'question_type' not in data:
            return jsonify({
                'error': 'Answer and question type are required'
            }), 400

        answer = data['answer']
        question_type = data['question_type']
       
        # Preprocess the answer
        preprocessed_answer = preprocess_text(answer)
       
        # Check plagiarism
        similarity_score, is_original = check_plagiarism(preprocessed_answer, question_type)
       
        # Calculate other metrics
        english_quality = calculate_english_quality(answer)
       
        # Calculate overall accuracy (example metric)
        accuracy = (0.7 if is_original else 0.3) + (english_quality * 0.3)
       
        # Save the submission
        save_submission(question_type, answer, similarity_score)
       
        response = {
            'is_original': bool(is_original),
            'similarity_score': float(similarity_score),
            'english_quality': float(english_quality),
            'accuracy': float(accuracy)
        }
       
        return jsonify(response)

    except Exception as e:
        print(f"Error in evaluate_answer: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred while evaluating the answer'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)