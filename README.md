
# What's Your Jam – Final Project

## Overview
This project builds a **music recommender system** using **machine learning**, trained on a dataset sourced from Kaggle and the **Spotify API**. Unlike traditional genre-based recommendations, this system suggests songs that **share similar characteristics**, even if they belong to different genres.

The approach centers around **K-Means clustering**, which groups songs based on musical attributes. Users can input a song, and the model will suggest **similar tracks**, provided the song exists in the dataset. To make the experience interactive, we implemented a **dashboard using Dash**, allowing users to explore and receive recommendations dynamically.

The dataset includes **over 232,000 songs** with multiple **audio features** that improve song similarity analysis beyond just genre.

## Features Used in the Model
These song attributes are crucial in understanding **musical similarities** beyond genre:
* **Popularity** – A song's ranking based on Spotify engagement
* **Acousticness** – Probability of a track being acoustic (0.0 to 1.0)
* **Danceability** – Suitability for dancing based on rhythm and tempo (0.0 to 1.0)
* **Duration (ms)** – Track length in milliseconds
* **Energy** – Intensity/activity level (0.0 to 1.0)
* **Instrumentalness** – Likelihood of a song being purely instrumental
* **Key** – The musical key the song is in
* **Liveness** – Detects the presence of a live audience in a track
* **Loudness** – Overall volume level in decibels
* **Mode** – Musical mode (Major or Minor)
* **Speechiness** – Measures spoken content (e.g., rap vs. instrumental)
* **Tempo** – Beats per minute (BPM) of the track
* **Time Signature** – The number of beats per measure
* **Valence** – Measures positivity (happy vs. melancholic music)

Each of these features contributes to **clustering songs effectively** and enhancing recommendations.

## How the Model Works
1. **Feature Extraction & Preprocessing** – The dataset is cleaned, missing values are handled, and numerical/categorical features are encoded. **Principal Component Analysis (PCA)** is used to simplify the data by reducing dimensionality before clustering.
2. **K-Means Clustering** – Once features are processed, **K-Means groups songs** based on musical attributes to create meaningful clusters. This allows songs to be grouped according to **similar audio characteristics** rather than just genre labels.
3. **Genre Classification Using Random Forest** – **Random Forest** is used to classify songs into genres, making predictions based on key song features such as danceability, energy, tempo, and valence. Since it is a supervised learning model, it requires labeled data to learn patterns and make accurate predictions.
4. **Recommendation Engine** – When a user selects a song, the system retrieves **musically similar tracks** from the pre-trained clusters and genre classification results.
5. **Dashboard Implementation** – Users interact with the model via a **Dash-powered web interface**, enabling song exploration and recommendations dynamically.

## Machine Learning Techniques Used
We are using **Random Forest, PCA, and K-Means** because each method serves a distinct purpose in the project.
* **Random Forest** is used for **genre classification**, predicting which genre a song belongs to based on key musical features.
* **PCA (Principal Component Analysis)** assists in **dimensionality reduction**, simplifying complex data while retaining important variance.
* **K-Means Clustering** is responsible for **grouping similar songs**, helping generate personalized recommendations and playlists.

## How They Work Together
1. **PCA & K-Means are applied for clustering**, identifying relationships between songs.
2. **Random Forest is trained on the original (scaled) dataset**, ensuring accurate genre classification.
3. **Cluster labels from K-Means can be used as additional features** in the classifier to refine predictions.

This approach ensures **songs are classified effectively while also being grouped into meaningful clusters**, improving recommendations and analysis.

## Team Members
* **Forrest Margulies**
* **Gurpreet Badrain**
* **Luis Lopez**
* **Yue Deng**
