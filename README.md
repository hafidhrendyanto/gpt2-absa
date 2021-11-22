# Aspect Based Sentiment Analysis using OpenAI's GPT-2
Currently, most sentiment analysis solution are using BERT as a base for it's model and only a few solution use Open AI's GPT-2.
This solution use OpenAI's GPT-2 to solve Aspect Based Sentiment Analysis (ABSA) problem with less than 150 lines of code and 
no changes to the model architecture by reframing the problem as a language generation task which GPT-2 excell.

This solution achieve 0.925 F1 score for *sentiment polarity classification* task on SemEval 2016 Task 5 Restaurant dataset.

I wrote this solution for my undergrade thesis and I hope that this solution can help you make a your next best project or inspire other research in this topic. Thank you.
Any input is very appreciated

# Try it yourself
You can try the solution by cloning the repo to your local computer and running the main.py or sample.ipynb
I recomend running sample.ipynb on google colab [here](https://colab.research.google.com/github/hafidhrendyanto/gpt2-absa/blob/main/sample.ipynb)

# Installation
You can install the package on your local computer by running
`pip install git+https://github.com/hafidhrendyanto/gpt2-absa.git`