import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import difflib
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
df = pd.read_csv("resources/SpotifyFeatures.csv")

# --- Data Preprocessing ---

# Convert 'mode' from categorical to numeric
df["mode"] = df["mode"].map({"Major": 1, "Minor": 0})

# Standardize genre names
df["genre"] = df["genre"].replace("Children’s Music", "Children's Music")
# Remove genres you don't want; example:
# df = df[df["genre"] != "A Capella"]

# Encode genre to numeric
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["genre"])

# Encode artist names
artist_encoder = LabelEncoder()
df["artist_encoded"] = artist_encoder.fit_transform(df["artist_name"])

# Check all required features exist in DataFrame
numeric_features = [
    "danceability", "energy", "tempo", "acousticness",
    "instrumentalness", "valence", "loudness", "speechiness",
    "artist_encoded"  # Include artist encoding
]

important_features = ["danceability", "energy", "tempo", "acousticness",
                    "instrumentalness", "valence", "loudness", "speechiness",
                    "artist_encoded", "song_length", "key", "genre_encoded"]

# Make sure all features are present
missing_cols = [col for col in numeric_features if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# --- Filter dataset to remove problematic rows ---
# Example: remove rare genres or other unwanted data
df = df[df["genre"] != "A Capella"]  # Remove rare genre
# (Apply other filters here if needed)

# --- Sample dataset for modeling ---
# Ensure sample size does not exceed dataset size
sample_size = min(5000, len(df))
df_sample = df.sample(sample_size, random_state=42).reset_index(drop=True)

# --- Features scaling ---
scaler = StandardScaler()
X_scaled_df = pd.DataFrame(scaler.fit_transform(df_sample[important_features]), columns=important_features)
X = X_scaled_df 

# --- Prepare target variable for classification ---
y = df_sample["genre_encoded"].copy()

# Ensure features_no_nan is correctly assigned, matching your features used
features_no_nan = important_features

# Verify lengths match
assert len(X_scaled_df.columns) == len(features_no_nan), "Feature list length mismatch!"
assert len(X_scaled_df) == len(y), "X and y length mismatch!"

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

# Feature importance
rf_temp = RandomForestClassifier(n_estimators=100)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": features_no_nan, "Importance": importances})
feature_importance_df.sort_values("Importance", ascending=False, inplace=True)

# Apply PCA(Principal Component Analysis) for Dimensionality Reduction
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_scaled_df)
#print(f"Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.2f}")  # Check variance retention

# Create DataFrame for PCA results
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"])
pca_df["artist_name"] = df_sample["artist_name"].values
pca_df["original_index"] = df_sample.index  # Keep original index for mapping clusters
#pca_df.head()  # Preview PCA DataFrame

# Silhouette Score for Clustering
for k in range(2, 10):  # Avoid k=1 (single cluster isn't useful)
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    #print(f"Clusters: {k}, Silhouette Score: {score:.4f}")

# Elbow Method to Determine Optimal Number of Clusters
# The Elbow Method helps to find the optimal number of clusters by plotting the inertia (sum of squared distances)
# Define range of clusters to test
num_clusters = range(1, 10)  # Testing 1 to 10 clusters
inertia_values = []

for k in num_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia_values.append(kmeans.inertia_)  # Store inertia for each cluster count

# Davies-Bouldin Index (DBI) is another useful metric to evaluate clustering quality.
# Compute DBI for different cluster numbers (K=2 to K=9)
for k in range(2, 10):  
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)  # Fit K-Means clustering
    dbi_score = davies_bouldin_score(X_pca, labels)  # Compute DBI
    #print(f"Clusters: {k}, Davies-Bouldin Score: {dbi_score:.4f}")

# Perform K-Means Clustering
# Group similar songs based on their PCA-transformed features.
kmeans = KMeans(n_clusters= 2, random_state=42, n_init=10)
# 
df_sample["Cluster"] = kmeans.fit_predict(X_pca)
# Merge Cluster Labels into PCA DataFrame
pca_df["Cluster"] = df_sample.loc[pca_df["original_index"], "Cluster"].values

# Define PCA components to plot
pca_pairs = [("PC1", "PC2"), ("PC2", "PC3"), ("PC3", "PC4"), ("PC4", "PC5"), ("PC5", "PC6"), ("PC6", "PC7")]

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (pc_x, pc_y) in enumerate(pca_pairs):
    ax = axes[i]
    
    # Scatter plot of PCA clusters
    sns.scatterplot(
        x=pca_df[pc_x], 
        y=pca_df[pc_y], 
        hue=pca_df["Cluster"].astype(str), 
        palette="viridis", 
        alpha=0.7, 
        ax=ax
    )
    
    # Compute centroids
    centroids = pca_df.groupby("Cluster")[[pc_x, pc_y]].mean()
    
    # Extract cluster colors from the scatter plot
    cluster_palette = dict(zip(pca_df["Cluster"].unique(), sns.color_palette("viridis", len(pca_df["Cluster"].unique()))))

# Normalize text (removes accents/diacritics)
def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()

# Clean and standardize text
def clean_text(text):
    return (
        text.lower()
        .strip()
        .replace('"', '')
        .replace("'", "")
        .replace("’", "'")
        .replace(".", "")
        .replace(":", "")
        .strip()
    )

# Combined cleaning: first normalize, then clean punctuation
def full_clean_text(text):
    return clean_text(normalize_text(text))

# Standardize track names and artist names
df["track_name"] = df["track_name"].astype(str).apply(full_clean_text)
df["artist_name"] = df["artist_name"].astype(str).str.lower().str.strip()

def verify_song_in_data(input_song):
    """Returns DataFrame rows matching the cleaned input song."""
    cleaned_input = full_clean_text(input_song)
    return df[df["track_name"].str.contains(cleaned_input, case=False, na=False)]

def find_closest_match(input_song, cutoff=0.9, num_matches=5):
    """
    Returns multiple closest matching songs (as lists) and their artists.
    - If an exact match exists, returns it as a single-element list.
    - Otherwise, returns possible close matches.
    Duplicate song titles are filtered out.
    """
    input_song_cleaned = full_clean_text(input_song)
    all_songs = df["track_name"].dropna().tolist()
    # Remove duplicates while preserving order.
    unique_song_list = list(dict.fromkeys(all_songs))
    
    # If the artist is provided, attempt an exact match on both song and artist.
    if input_artist:
        input_artist_cleaned = full_clean_text(input_artist)
        exact_matches_df = df[
            (df["track_name"].str.fullmatch(input_song_cleaned, case=False, na=False)) &
            (df["artist_name"].str.fullmatch(input_artist_cleaned, case=False, na=False))
        ]
        if not exact_matches_df.empty:
            matched_songs = exact_matches_df["track_name"].tolist()
            matched_artists = exact_matches_df["artist_name"].tolist()
            # Deduplicate the (song, artist) pairs while preserving order.
            unique_matches = []
            for pair in zip(matched_songs, matched_artists):
                if pair not in unique_matches:
                    unique_matches.append(pair)
            for song, art in unique_matches:
                print(f"Exact match found: '{song}' by {art}")
            # Unzip unique_matches back into two lists.
            return [pair[0] for pair in unique_matches], [pair[1] for pair in unique_matches]
        else:
            print("Exact match on both song and artist not found. Searching by song name only:")
    
    # Search by song name only.
    exact_matches_song_df = df[df["track_name"].str.fullmatch(input_song_cleaned, case=False, na=False)]
    if not exact_matches_song_df.empty:
        matched_songs = exact_matches_song_df["track_name"].tolist()
        matched_artists = exact_matches_song_df["artist_name"].tolist()
        unique_matches = []
        for pair in zip(matched_songs, matched_artists):
            if pair not in unique_matches:
                unique_matches.append(pair)
        for song, art in unique_matches:
            print(f"Exact match found: '{song}' by {art}")
        return [pair[0] for pair in unique_matches], [pair[1] for pair in unique_matches]
    
    # Fuzzy matching if no exact song match is found.
    all_songs = df["track_name"].dropna().tolist()  # include duplicates here
    closest_matches = difflib.get_close_matches(input_song_cleaned, all_songs, n=num_matches, cutoff=cutoff)
    if closest_matches:
        matched_songs, matched_artists = [], []
        print("Possible close matches:")
        for song in closest_matches:
            song_rows = df[df["track_name"].str.fullmatch(song, case=False, na=False)]
            for _, row in song_rows.iterrows():
                matched_songs.append(row["track_name"])
                matched_artists.append(row["artist_name"])
        # Deduplicate matches.
        unique_matches = []
        for pair in zip(matched_songs, matched_artists):
            if pair not in unique_matches:
                unique_matches.append(pair)
        for song, art in unique_matches:
            print(f"Exact match found: '{song}' by {art}")
        return [pair[0] for pair in unique_matches], [pair[1] for pair in unique_matches]
    else:
        print(f"No matches found for '{input_song}'" + (f" by '{input_artist}'." if input_artist else "."))
        return [], []

# Recommend songs based on similarity

def recommend_song(input_song, input_artist=None, cutoff=0.995, similarity_threshold=0.5, pre_matched=False):
    if not pre_matched:
        # First, try to find an exact or close match
        matched_song, matched_artist = find_closest_match(input_song, cutoff)
        if isinstance(matched_song, list) and len(matched_song) > 0:
            # Use the first match
            matched_song = matched_song[0]
            matched_artist = matched_artist[0]
        else:
            # No match found; find the closest song in dataset
            all_songs = df['track_name'].dropna().tolist()
            all_cleaned = [full_clean_text(s) for s in all_songs]
            input_cleaned = full_clean_text(input_song)
            import difflib
            close_matches = difflib.get_close_matches(input_cleaned, all_cleaned, n=1, cutoff=0.75)
            if close_matches:
                idx = all_cleaned.index(close_matches[0])
                matched_song = all_songs[idx]
                # Get artist for that song
                artist_series = df[df['track_name'] == matched_song]['artist_name']
                matched_artist = artist_series.iloc[0] if not artist_series.empty else "Unknown"
                # **Return info and let caller decide** (e.g., prompt user)
                return {
                    'type': 'fallback',
                    'matched_song': matched_song,
                    'matched_artist': matched_artist,
                    'message': f"Closest match found: '{matched_song}' by '{matched_artist}'."
                }
            else:
                # No close match: fallback to input
                matched_song = input_song
                matched_artist = input_artist
                return {
                    'type': 'no_match',
                    'matched_song': matched_song,
                    'matched_artist': matched_artist,
                    'message': f"No close matches found for '{input_song}'. Proceeding with original input."
                }
    else:
        # Caller forced pre_matched=True, meaning it already has a valid song/artist
        matched_song = input_song
        matched_artist = input_artist

    # Now, proceed to generate recommendations based on matched_song
    matched_idx = df[df["track_name"] == matched_song].index
    if len(matched_idx) == 0:
        return f"Matched song '{matched_song}' not found in dataset."

    # Get features and compute similarity
    numeric_features = ["danceability", "energy", "valence", "tempo", "speechiness"]
    song_features = df.loc[matched_idx[0], numeric_features].values.reshape(1, -1)
    valid_rows = df.dropna(subset=numeric_features).index
    df_valid = df.loc[valid_rows, numeric_features]
    similarities = cosine_similarity(song_features, df_valid)
    similarity_scores = similarities[0] / similarities[0].max()

    # Prepare recommendations DataFrame
    df_filtered = df.loc[valid_rows].copy()
    df_filtered["Similarity"] = similarity_scores
    artist_for_exclusion = input_artist if input_artist else matched_artist

    # Filter out same artist
    recommendations = df_filtered[
        (df_filtered["artist_name"] != artist_for_exclusion) &
        (df_filtered["Similarity"] >= max(similarity_threshold, df_filtered["Similarity"].mean()))
    ]

    # Deduplicate
    recommendations = recommendations.drop_duplicates(subset=["track_name", "artist_name"])

    # Filter by genre if column exists
    if "genre" in df.columns:
        input_genre = df.loc[df["track_name"] == matched_song, "genre"].values[0]
        recommendations = recommendations[recommendations["genre"] == input_genre]

    # Final sort & top 10
    recommendations = recommendations.sort_values(
        by=["Similarity", "danceability", "energy"], ascending=[False, False, False]
    ).head(10)

    recommendations_list = [
        f"{row['track_name']} by {row['artist_name']}"
        for _, row in recommendations.iterrows()
    ]

    # Return recommendations
    return {
        'type': 'recommendations',
        'recommendations': recommendations_list,
        'message': None
    }

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
# # Load banner image (base64)
# image_path = os.path.join(os.getcwd(), 'assets', 'banner.png')
# app.title = "Music Recommender"
# if os.path.exists(image_path):
#     with open(image_path, 'rb') as img_file:
#         encoded_image = base64.b64encode(img_file.read()).decode()
#     img_component = html.Img(
#         src=f'data:image/png;base64,{encoded_image}',
#         style={
#             'width': '100%',
#             'height': 'auto',
#             'marginBottom': '10px'
#         }
#     )
# else:
#     img_component = html.Div("Banner image not found.", style={'color': 'red', 'marginBottom': '10px'})

app.layout = dbc.Container([
    # img_component,
    html.H1("🎵 Music Recommender 🎵", className="text-center my-4"),
    
    # Tabs structure
    dcc.Tabs(id="tabs", value='tab-recommender', children=[
        dcc.Tab(label='Music Recommender', value='tab-recommender'),
        dcc.Tab(label='Spotify Dataset EDA', value='tab-eda'),
    ]),
    html.Div(id='tab-content', style={
        "position": "relative",
        "zIndex": "1",
        "padding": "20px",
        "backgroundColor": "rgba(255,255,255,0.8)"
    })
], fluid=True)

# Callback to switch between tabs
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-recommender':
        return html.Div([
            # Inputs side-by-side
            dbc.Row([
                dbc.Col([
                    dbc.Label("Song Name"),
                    dbc.Input(id="song_input", placeholder="Enter song name", type="text"),
                ], width=6),
                dbc.Col([
                    dbc.Label("Artist Name"),
                    dbc.Input(id="artist_input", placeholder="Enter artist name", type="text"),
                ], width=6),
            ], className="mb-4"),
            
            # Button below the inputs, centered
            dbc.Row([
                dbc.Col([
                    dbc.Button("Get Recommendations", id="recommend_button", color="primary", size="lg")
                ], width={"size": 3, "offset": 2}),  # offset to center
            ], className="mb-4"),
            
            # Output area for recommendations
            html.Div(id="recommendations", style={
                "marginTop": "20px",
                "padding": "15px",
                "border": "1px solid #ccc",
                "borderRadius": "8px",
                "backgroundColor": "#f9f9f9"
            })
        ])
    elif tab == 'tab-eda':
        return html.Div([
        html.H4("Spotify Dataset EDA Coming Soon"),
        # Adding carousel
        # dbc.Carousel(
        #     items=[
        #         {"key": "1", "src": "assets/Boston-cream-donut.png", "caption": "This is a Boston cream donut", "img_style": {"max-height": "500px"}},
        #         {"key": "2", "src": "assets/assets/f8b76d1883e09d74.png", "header": "Minecraft skin", "caption": "This is my Minecraft skin", "img_style": {"max-height": "500px"}}
        #     ]
        # )
    ])

# Callback for song recommendations
@app.callback(
    Output("recommendations", "children"),
    [Input("recommend_button", "n_clicks")],
    [State("song_input", "value"),
    State("artist_input", "value")]
)
def update_recommendations(n_clicks, song_name, artist_name):
    if n_clicks and song_name:
        recommendations = recommend_song(song_name, artist_name)
        if isinstance(recommendations, list) and recommendations:
            return [html.Div(f"{i+1}. {rec}") for i, rec in enumerate(recommendations)]
        else:
            return html.Div(recommendations)
    return ""


if __name__ == '__main__':
    app.run(debug=True)