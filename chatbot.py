import re
import random
class DataScienceChatbot:
  ### potential Negative Responses
   negative_responses = ("no","nope","nah","not a chance","sorry")
  ### Exit coversations keywords
   exit_commands = ("quit","exit","bye","later","stop","end","goodbye")
  ### Random starter questions
   random_questions = (
      "What is Data Science?",
      "What is Artificial Intelligence?",
      "What is Machine Learning?",
      "What is Deep Learning?",
      "What is Big Data?",
      "What is the difference between AI and Data Science?",
      "What is data scientist?",
      "What is Natural Language Processing (NLP)?",
      "What jobs can I get with Data Science skills?",
      "What tools are popular in Data Science?",
      "What is supervised learning?",
      "What is unsupervised learning?",
      "What is reinforcement learning?",
      "What is a neural network?",
      "What is the difference between AI, ML, and DL?",
        "What is model training?",
        "What is model evaluation?",
        "What is overfitting?",
        "What is underfitting?",
        "What is cross-validation?",
        "What is a dataset?",
        "What is data cleaning?",
        "What is feature engineering?",
          "What is data preprocessing?",
            "What is data visualization?",
            "What is dimensionality reduction?",
              "What is data wrangling?",
              "What is data annotation?",
              "What are structured and unstructured data?",
              "What is a data pipeline?",
              "What is regression?",
              "What is classification?",
              "What is clustering?",
              "What is decision tree?",
              "What is random forest?",
             "What is support vector machine (SVM)?",
              "What is k-nearest neighbors (KNN)?",
              "What is k-means clustering?",
              "What is gradient descent?",
              "What is logistic regression?",
              "What is deep reinforcement learning?",
              "What is transfer learning?",
                "What is a convolutional neural network (CNN)?",
                "What is a recurrent neural network (RNN)?",
                  "What is natural language generation (NLG)?",
                  "What is computer vision?",
                  "What is sentiment analysis?",
                  "What are large language models (LLMs)?",
                  "What is the Turing Test?"
   )
def __init__(self):
        self.name = "DataScienceBot"
        print(" Hello! I'm your Data Science & AI chatbot.")
        print("You can ask me questions about Data Science, AI, and Machine Learning.")
        print("Type 'exit' or 'quit' anytime to leave.\n")
        self.chat()   # Start the conversation automatically
def start_chat(self):
        print(f"{self.name}: Hello! Ask me something about Data Science or AI. (Type 'exit' to quit)")
        
        while True:
            user_input = input("You: ").lower()
            
            # Exit condition
            if user_input in self.exit_commands:
                print(f"{self.name}: Goodbye 👋 Keep learning Data Science & AI!")
                break
            
            # Negative responses
            elif user_input in self.negative_responses:
                print(f"{self.name}: No worries! Maybe ask me another question?")
            
            # Random question fallback
            else:
                import random
                question = random.choice(self.random_questions)
                print(f"{self.name}: Hmm… try this one→{question}")
# Run chatbot
if __name__ == "_main_":
    bot = DataScienceChatbot()
    bot.start_chat()
    