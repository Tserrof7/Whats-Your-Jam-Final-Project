Spotify Recommendation Dashboard

A lightweight Streamlit dashboard that displays song recommendations powered by a recommendation engine built in a Jupyter Notebook. The app integrates with the Spotify API (via Spotipy) so that users can search for and play tracks on an active Spotify device.

Overview

Recommendation Engine: Developed in a Jupyter Notebook using text normalization, fuzzy matching, and cosine similarity on Spotify's features.
Dashboard: Exports recommendations to a CSV file (in the Output folder) and loads them into a Streamlit app for an interactive view.
Spotify Integration: Uses Spotify OAuth for authentication and Spotipy for controlling playback.
Requirements

Python: 3.10 (developed in a conda dev environment)
Packages: streamlit, spotipy, pandas, scikit-learn, etc.
(A complete list is provided in requirements.txt)
Installation

Clone the Repository:

git clone https://github.com/Tserrof7/Whats-Your-Jam-Final-Project.git
cd Whats-Your-Jam-Final-Project
Set Up Environment: If using Conda:

conda create -n dev python=3.10
conda activate dev
pip install -r requirements.txt
Spotify API Setup

Go to the Spotify Developer Dashboard, create (or select) an app, and copy your Client ID and Client Secret.
Redirect URI: Use an explicit loopback address, e.g.,
REDIRECT_URI = "http://127.0.0.1:7777/callback"
Add this URI to your app's settings on Spotify.
Usage

Generate Recommendations: Run the Jupyter Notebook (e.g., spotify.ipynb) to generate and export recommendations as Output/recommended_songs.csv.
Launch the Dashboard:
streamlit run dashboard.py
Ensure you have an active Spotify device ready, then use the Play buttons to trigger playback.
Project Structure

├── dashboard.py                # Streamlit dashboard
├── spotify.ipynb               # Notebook with recommendation engine
├── Output/recommended_songs.csv # Exported recommendations
├── resources/SpotifyFeatures.csv # Dataset for recommendations
├── requirements.txt            # Dependencies
└── README.md                   # This file
Features

Interactive Dashboard: Browse recommended songs with detailed track information
Spotify Playback Control: Play tracks directly from the dashboard on your active Spotify device
Data-Driven Recommendations: Leverages audio features and text similarity for intelligent song suggestions
Easy Setup: Simple configuration with Spotify API credentials
How It Works

The recommendation engine analyzes song features from the Spotify dataset
Uses cosine similarity and fuzzy matching to find similar tracks
Exports recommendations to a CSV file for the dashboard to consume
The Streamlit app provides an intuitive interface to explore and play recommendations
Contributing

This project is based on the original repository: Whats-Your-Jam-Final-Project
