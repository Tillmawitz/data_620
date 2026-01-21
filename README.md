# DATA 620 - Web Analytics

Coursework for the CUNY SPS Master's course which focussed on network analysis and natural language processing. Notable projects are:

1. [The Final Project](final/Tillmawitz_data_620_final.ipynb) - Evaluating Text Encoding Methods

    This project conducts an experiment that compares TF-IDF text encoding to modern sentence transformers by encoding texts from the Newsgroup Dataset and attempting to properly classify texts. After encoding the texts, a graph is constructed using KNN and cosine similarity with PyTorch Geometric and NetworkX. Several community detection methods are then applied to the graphs and the resulting performance is analyzed to evaluate the efficacy of the different encoding methods.

2. [Project 2](Tillmawitz_project_2.ipynb) - arXiv Citation Network Analysis

    A project analysing a citation network of academic papers consisting of over 400,000 nodes. Deals with analysis of a large bipartite network and extracting meaningful information using NetworkX.

3. [Week 10](Tillmawitz_week_10.ipynb) - Document Classification

    A comparison of Logistic Regression, Random Forest, Adaboost, and an MLP on a spam detection problem using primarily PyTorch and scikit-learn.

4. [Week 6](Tillmawitz_week_6.ipynb) - Southern Women Analysis

    An analysis of the classic Southern Women community detection problem. Notable for the visualizations and as an easily understandable analysis of bipartite graphs.