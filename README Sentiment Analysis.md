Sentiment Analysis
Project Overview
The Sentiment Analysis component of InsightX explores the use of deep learning models to classify the sentiment of text data, specifically from Twitter. The project focuses on the following models:

Long Short-Term Memory (LSTM) Network
Gated Recurrent Unit (GRU) Network
Bidirectional LSTM
Model Descriptions
1. Long Short-Term Memory (LSTM) Network
Objective: Capture long-term dependencies in the text data to improve sentiment classification accuracy.
Architecture:
Embedding layer: Converts words into dense vectors of fixed size.
LSTM layers: Capture temporal dependencies in the data.
Dense output layer: Uses sigmoid activation for binary classification.
Outcome: The LSTM model performed exceptionally well, showing its capability in handling sequences of text data and providing accurate sentiment predictions.
2. Gated Recurrent Unit (GRU) Network
Objective: Provide a simpler and faster alternative to LSTM with comparable performance.
Architecture:
Embedding layer: Similar to LSTM.
GRU layers: Fewer parameters compared to LSTM, but still capture temporal dependencies.
Dense output layer: Uses sigmoid activation.
Outcome: The GRU model, while faster, underperformed compared to LSTM, particularly in capturing the nuances of sentiment in longer text sequences.
3. Bidirectional LSTM
Objective: Improve sentiment analysis by considering the context from both directions in a text sequence.
Architecture:
Embedding layer: Converts words into vectors.
Bidirectional LSTM layers: Process the text in both forward and backward directions.
Dense output layer: Uses sigmoid activation.
Outcome: The Bidirectional LSTM outperformed the standard LSTM, making it the most accurate model for sentiment classification in this project.
Hyperparameter Tuning and Fine-Tuning
For each model, hyperparameter tuning was performed to optimize:

Learning rate: Critical for model convergence.
Number of layers and units: Adjusted to balance performance and training time.
Activation functions: ReLU and sigmoid were primarily used, with experiments involving tanh for comparison.
Additionally, models were fine-tuned by experimenting with different activation functions, dropout rates, and batch sizes. These adjustments were crucial in achieving the final performance metrics.

Testing and Evaluation
After training on Twitter data, the models were tested on sentiment data from Reddit to evaluate their generalizability. The LSTM model demonstrated superior performance in text summarization and sentiment prediction, while the GRU model struggled with longer and more complex sentences, confirming the LSTM's robustness.

Visualization with Tableau
A Tableau dashboard was created to visually compare the performance of the models. The dashboard includes:

Accuracy comparison: A bar chart comparing the accuracy of LSTM, GRU, and Bidirectional LSTM models.
Confusion matrices: Visual representations of the prediction errors for each model.
Sentiment distribution: Pie charts showing the distribution of predicted sentiments across the test data.
Real-World Application
Sentiment analysis has broad applications, including:

Customer Feedback Analysis: Understanding customer sentiment from reviews or social media.
Brand Monitoring: Tracking public sentiment towards a brand.
Market Research: Analyzing trends and public opinion on various topics.
Results and Conclusion
The LSTM-based models, particularly the Bidirectional LSTM, outperformed others, proving their effectiveness for sentiment analysis tasks. The Tableau dashboard provides a clear visualization of these results, making it easier for stakeholders to interpret the findings.
