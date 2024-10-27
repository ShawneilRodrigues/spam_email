
# Email Spam Classifier

This project is an end-to-end machine learning deployment of an Email Spam Classifier, which identifies whether an incoming email is spam or not. The deployed model is accessible in live production at [https://spam-email-4.onrender.com](https://spam-email-4.onrender.com).

## Project Overview

The Email Spam Classifier is a machine learning-based system trained on labeled email data to differentiate between spam and non-spam (ham) emails. Using various natural language processing (NLP) techniques, the model processes email content and makes predictions in real time.

### Key Features:

1. **Machine Learning Pipeline**: 
   - Preprocessing of raw text data including tokenization, stemming/lemmatization, and vectorization using TF-IDF.
   - A classification model trained to predict whether an email is spam based on textual features.
   
2. **Model Selection**: 
   - Models considered include Logistic Regression, Naive Bayes, Support Vector Machines (SVM), and Random Forest.
   - The final model was chosen based on precision, recall, and F1-score, providing the best balance for handling spam detection in real-world scenarios.
   
3. **Live Deployment**: 
   - The model is deployed in a production environment using [Render](https://render.com), ensuring scalability and real-time predictions.
   - The deployment is done through a web interface, allowing users to input emails and receive instant classification results.

## Technologies Used

- **Python**: Core programming language for data preprocessing, model training, and API development.
- **Scikit-learn**: Used for building and training the machine learning models.
- **Flask**: A lightweight web framework to serve the model through an API.
- **NLP Libraries**: NLTK and Scikit-learn's `CountVectorizer` and `TfidfTransformer` are used for text processing.
- **Render**: For hosting the web app, ensuring live production readiness and scalability.

## How It Works

1. **Input**: Users submit the content of an email they want to classify.
2. **Processing**: The email is preprocessed, including tokenization, removing stop words, and applying TF-IDF transformation.
3. **Prediction**: The preprocessed email is passed through the trained model, which outputs whether it is classified as "spam" or "ham".
4. **Output**: The result is displayed to the user along with a confidence score.

## Live Demo

You can try out the live spam classification demo here: [https://spam-email-4.onrender.com](https://spam-email-4.onrender.com)

Simply enter the text of an email in the provided form and click submit to see whether the model detects it as spam.

## Conclusion

This Email Spam Classifier demonstrates the power of machine learning in automating tasks such as email filtering. The deployment ensures that the model is ready for real-world usage, scalable for multiple requests, and able to provide predictions with high accuracy.

