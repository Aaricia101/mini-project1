import os
import sys
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

SIMILARITY_THRESHOLD = 0.55
ESCALATION_CONFIDENCE = 0.90

class StudentSupportAI:
    def __init__(self, knowledge_base_file="knowledge_base.csv"):
        self.knowledge_base_file = knowledge_base_file
        self.questions = []
        self.answers = []
        self.question_embeddings = None
        self.chat_history = []

        print("Loading Student Support AI...")

        self.load_knowledge_base()
        self.load_models()
        self.create_question_embeddings()

        print("System ready.\n")

    def load_knowledge_base(self):
        try:
            if not os.path.exists(self.knowledge_base_file):
                print(f"Error: {self.knowledge_base_file} was not found.")
                sys.exit()

            data = pd.read_csv(self.knowledge_base_file)

            if "question" not in data.columns or "answer" not in data.columns:
                print("Error: CSV file must contain 'question' and 'answer' columns.")
                sys.exit()

            self.questions = data["question"].astype(str).tolist()
            self.answers = data["answer"].astype(str).tolist()

            print(f"Loaded {len(self.questions)} questions from the knowledge base.")

        except Exception as error:
            print("There was an error loading the knowledge base.")
            print(error)
            sys.exit()

    def load_models(self):
        try:
            print("Loading sentence embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            print("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )

        except Exception as error:
            print("There was an error loading the AI models.")
            print(error)
            sys.exit()

    def create_question_embeddings(self):
        try:
            print("Creating question embeddings...")
            self.question_embeddings = self.embedding_model.encode(self.questions)

        except Exception as error:
            print("There was an error creating embeddings.")
            print(error)
            sys.exit()

    def find_best_answer(self, user_question):
        user_embedding = self.embedding_model.encode([user_question])

        similarities = cosine_similarity(
            user_embedding,
            self.question_embeddings
        )

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[0][best_index])

        if best_score < SIMILARITY_THRESHOLD:
            return (
                "Sorry, I could not find a revelant answer, please contact an advisor.",
                best_score
            )

        return self.answers[best_index], best_score

    def analyze_sentiment(self, user_text):
        result = self.sentiment_analyzer(user_text)[0]

        label = result["label"].upper()
        score = float(result["score"])

        return label, score

    def should_escalate(self, user_text, sentiment_label, confidence_score):
        user_text = user_text.lower()

        return (
            sentiment_label == "NEGATIVE"
            and confidence_score > ESCALATION_CONFIDENCE         
        )

    def answer_user(self, user_input):
        

        if(user_input.strip() =="-1"):
            for history in self.chat_history:
                print("\n--- History Entry ---")
                print(f"User: {history['user']}")
                print(f"Sentiment: {history['sentiment']} ({history['confidence']:.2f})")
                print(f"Answer: {history['answer']}")
                print(f"Similarity: {history['similarity']:.2f}")
                print("---------------------\n")
        else:
            sentiment_label, confidence_score = self.analyze_sentiment(user_input)
            answer, similarity_score = self.find_best_answer(user_input)

            self.chat_history.append({
                "user": user_input,
                "sentiment": sentiment_label,
                "confidence": confidence_score,
                "answer": answer,
                "similarity": similarity_score
            })
            print(f"Sentiment: {sentiment_label} ({confidence_score:.2f})")

            if self.should_escalate(user_input, sentiment_label, confidence_score):
                print("Recommended escalation: Contact human advisor.")

            print(f"Answer: {answer}")
            print(f"Similarity score: {similarity_score:.2f}")
            print()

    def run(self):
        print("Welcome to Student Support AI")
        print("Type 'quit' to exit.")
        print()

        while True:
            print("Type '-1' to see the entire history.")
            user_input = input("You: ").strip()

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input == "":
                print("Please enter a question.")
                print()
                continue

            self.answer_user(user_input)


def main():
    assistant = StudentSupportAI("knowledge_base.csv")
    assistant.run()


if __name__ == "__main__":
    main()