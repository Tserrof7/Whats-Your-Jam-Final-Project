# --- Import Required Libraries ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import unicodedata
import difflib
import dash_bootstrap_components as dbc
import os
import base64
from dash import Dash, dcc, html, Input, Output, callback, State, ALL, callback_context
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Load Dataset
df = pd.read_csv("resources/SpotifyFeatures.csv")

# --- Data Preprocessing & Feature Selection ---
df["mode"] = df["mode"].map({"Major": 1, "Minor": 0})  # Convert Major/Minor to numeric
artist_encoder = LabelEncoder()
df["artist_encoded"] = artist_encoder.fit_transform(df["artist_name"])  # Encode artist names

# Convert categorical features to numeric
key_mapping = {k: i for i, k in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
df["key"] = df["key"].map(key_mapping).fillna(-1)  # Convert musical keys to numbers
df["time_signature"] = df["time_signature"].astype(str).str.extract(r'(\d+)').astype(float).fillna(df["time_signature"].mode()[0])  # Convert time_signature

# Compute rough song length metric
df["song_length"] = df["tempo"] * df["time_signature"]

# --- Speed Optimization: Use a Smaller Sample ---
df_sample = df.sample(3000, random_state=42).reset_index(drop=True)

# Encode genre labels numerically and ensure they start at 0
# Fit the encoder on the genres present in df_sample to avoid unseen labels
genre_encoder = LabelEncoder()
df_sample["genre_encoded"] = genre_encoder.fit_transform(df_sample["genre"])
df_sample["genre_encoded"] -= df_sample["genre_encoded"].min()  # Ensure labels start at 0

# Define important features for classification
important_features = ["danceability", "energy", "tempo", "acousticness",
                    "instrumentalness", "valence", "loudness", "speechiness",
                    "artist_encoded", "song_length", "key", "genre_encoded"]  # Removed mode & time_signature

# --- Handle Genre Issues ---
df_sample = df_sample[df_sample["genre"] != "A Capella"]  # Remove rare genre
df_sample["genre"] = df_sample["genre"].replace("Children’s Music", "Children's Music")  # Standardize genre names

# Remove columns with all NaN values
features_no_nan = [col for col in important_features if not df_sample[col].isna().all()]

# Scale Features
scaler = StandardScaler()
X_scaled_df = pd.DataFrame(scaler.fit_transform(df_sample[features_no_nan]), columns=features_no_nan)

# --- Genre Classification Model ---
X = X_scaled_df  # Features (scaled)
y = df_sample["genre_encoded"]  # Target labels (encoded genre)

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Importance Analysis ---
rf_temp = RandomForestClassifier(n_estimators=100)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": features_no_nan, "Importance": importances})
feature_importance_df.sort_values("Importance", ascending=False, inplace=True)

# --- Train Optimized Random Forest Model with Class Balancing ---
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5, class_weight="balanced", n_jobs=-1)  # Class weighting instead of SMOTE
rf_model.fit(X_train, y_train)

# Evaluate Random Forest Model
y_pred_rf = rf_model.predict(X_test)
# Apply PCA(Principal Component Analysis) for Dimensionality Reduction
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_scaled_df)

# Create DataFrame for PCA results
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"])
pca_df["artist_name"] = df_sample["artist_name"].values
pca_df["original_index"] = df_sample.index  # Keep original index for mapping clusters

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
# LEAST DBI indicates better clustering quality
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

# Compute UMAP embedding if not already done
umap_model = umap.UMAP(n_components=6, random_state=42)
umap_embedding = umap_model.fit_transform(X_scaled_df)

# Create DataFrame for UMAP results
umap_columns = [f"UMAP{i+1}" for i in range(umap_embedding.shape[1])]
umap_df = pd.DataFrame(umap_embedding, columns=umap_columns)
umap_df["Cluster"] = df_sample["Cluster"].values

# Define UMAP component pairs to visualize
umap_pairs = [("UMAP1", "UMAP2"), ("UMAP2", "UMAP3"), ("UMAP3", "UMAP4"), ("UMAP4", "UMAP5"), ("UMAP5", "UMAP6")]

# Create subplots dynamically based on number of pairs
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjusted figure size
axes = axes.flatten()

for i, (umap_x, umap_y) in enumerate(umap_pairs):
    ax = axes[i]

    # Scatter plot of UMAP clusters
    sns.scatterplot(
        x=umap_df[umap_x], 
        y=umap_df[umap_y], 
        hue=umap_df["Cluster"].astype(str), 
        palette="viridis", 
        alpha=0.7, 
        ax=ax
    )

    # Compute centroids
    centroids = umap_df.groupby("Cluster")[[umap_x, umap_y]].mean()

    # Extract cluster colors from the scatter plot
    cluster_palette = dict(zip(umap_df["Cluster"].unique(), sns.color_palette("viridis", len(umap_df["Cluster"].unique()))))

    # Plot centroids with matching cluster colors
    for cluster, row in centroids.iterrows():
        ax.scatter(row[umap_x], row[umap_y], marker="X", s=200, c=[cluster_palette[cluster]], edgecolors="black")

    ax.set_xlabel(umap_x)
    ax.set_ylabel(umap_y)
    ax.set_title(f"UMAP Clusters: {umap_x} vs. {umap_y}")

    # Reduce overlapping text labels
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=30)

# Remove extra empty subplot if needed
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# --- Text Normalization and Cleaning Functions ---
def normalize_text(text):
    text = str(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()

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

def full_clean_text(text):
    return clean_text(normalize_text(text))

# ---Adding Track name and song ---
df["track_name"] = df["track_name"].astype(str).apply(full_clean_text)
df["artist_name"] = df["artist_name"].astype(str).str.lower().str.strip()

# --- Matching Function with Deduplication ---
def find_closest_matches(input_song, input_artist=None, cutoff=0.9, num_matches=5):
    """
    Returns matching songs along with their artist names.
    If input_artist is provided, it first attempts to find exact matches
    on both the song and artist. If none are found, it falls back to match on
    the song name only. Duplicate matches are removed before printing.
    """
    input_song_cleaned = full_clean_text(input_song)
    
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
            print("Exact match on both song and artist not found. Please try another song.")
    
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

def recommend_song(input_song, input_artist=None, cutoff=0.995, similarity_threshold=0.5, pre_matched=False):
    if not pre_matched:
        # First, try to find an exact or close match
        matched_song, matched_artist = find_closest_matches(input_song, cutoff)
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
                    #'message': f"Closest match found."
                }
            else:
                # No close match: fallback to input
                matched_song = input_song
                matched_artist = input_artist
                return {
                    'type': 'no_match',
                    'matched_song': matched_song,
                    'matched_artist': matched_artist,
                    #'message': f"No close matches found for '{input_song}'. Proceeding with original input."
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
        'message': 'Check these out:',
        'recommendations': recommendations_list,
    }

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
# # Load banner image (base64)
image_path = os.path.join('resources', 'banner.png')
# Initialize empty variable
banner_img = None

if os.path.exists(image_path):
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode()

    banner_img = html.Img(
        src=f'data:image/png;base64,{encoded_image}',
        style={
            'width': '100%',
            'height': 'auto',
            'marginBottom': '1px'
        }
    )
else:
    banner_img = html.Div(
        "Banner image not found.",
        style={'color': 'red', 'marginBottom': '10px'}
    )

app.layout = dbc.Container([
    # img_component,
    banner_img,
    html.H1(" ", className="text-center my-4"),
    
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
                    dbc.Input(id="song_input", placeholder="Enter song name", type="text"),
                ], width=6),
                dbc.Col([
                    dbc.Input(id="artist_input", placeholder="Enter artist name", type="text"),
                ], width=6),
            ], className="mb-4"),
            
            # Button below the inputs, centered
            dbc.Row([
                dbc.Col([
                    dbc.Button("You may also like", id="recommend_button", color="info", size="lg", style={"color": "#333"})
                ], width={"size": 3}),
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
        html.H4(" "),
        #Adding carousel
        dbc.Carousel(
            items=[
                {"key": "1", "src": "assets/1.png", "img_style": {"max-width": "800px","max-height": "900px","width": "auto","height": "auto"}},
                {"key": "2", "src": "assets/2.png", "img_style": {"max-width": "800px","max-height": "900px","width": "auto","height": "auto"}}
            ],
        className="d-flex justify-content-center"
        )
    ])

# Callback for song recommendations
@app.callback(
    Output('recommendations', 'children'),
    [
        Input('recommend_button', 'n_clicks'),
        Input({'type': 'song-choice', 'index': ALL}, 'n_clicks'),
        Input('song_input', 'value'),
        Input('artist_input', 'value')
    ],
    [
        State({'type': 'song-choice', 'index': ALL}, 'id'),
        State('song_input', 'value'),
        State('artist_input', 'value')
    ],
    prevent_initial_call=True,
    allow_duplicate=True
)
def handle_all_triggers(n_clicks, song_choice_n_clicks, song_input, artist_input, song_choice_ids, song_input_state, artist_input_state):
    ctx = callback_context
    if not ctx.triggered:
        return ""
    trigger_prop = ctx.triggered[0]['prop_id']
    trigger_id = trigger_prop.split('.')[0]

    # **Clear output on input change**
    if trigger_id in ['song_input', 'artist_input']:
        return ""

    # **Handle user clicking a song option button**
    if 'song-choice' in trigger_id:
        for n, id_dict in zip(n_clicks, ids):
            if n and n > 0:
                selected_song = id_dict['index']
                recs_result = recommend_song(selected_song)
                if isinstance(recs_result, dict):
                    if recs_result.get('type') == 'recommendations':
                        recs = recs_result.get('recommendations', [])
                        return html.Div([
                            html.H4("Recommendations:"),
                            *[html.Div(f"{i+1}. {rec}") for i, rec in enumerate(recs)]
                        ])
        return "No recommendations available."

    # **Main button clicked**
    if n_clicks:
        # Your matching logic here
        # For brevity, let's assume the matching logic you provided
        input_song_cleaned = full_clean_text(song_input)
        input_artist_cleaned = full_clean_text(artist_input) if artist_input else None

        # 1. Exact match on artist+song
        if artist_input:
            exact_match_df = df[
                (df["track_name"].str.fullmatch(input_song_cleaned, case=False, na=False)) &
                (df["artist_name"].str.fullmatch(input_artist_cleaned, case=False, na=False))
            ]
            if not exact_match_df.empty:
                row = exact_match_df.iloc[0]
                matched_song = row["track_name"]
                matched_artist = row["artist_name"]
                recs_result = recommend_song(matched_song, matched_artist)
                if isinstance(recs_result, dict):
                    if recs_result.get('type') == 'recommendations':
                        recs = recs_result.get('recommendations', [])
                        return html.Div([
                            html.H4("Recommendations:"),
                            *[html.Div(f"{i+1}. {rec}") for i, rec in enumerate(recs)]
                        ])

        # 2. List options for exact match on song name only
        exact_songs_df = df[df["track_name"].str.fullmatch(input_song_cleaned, case=False, na=False)]
        if not exact_songs_df.empty:
            options = []
            for _, row in exact_songs_df.iterrows():
                options.append(
                    html.Button(
                        f"{row['track_name']} by {row['artist_name']}",
                        id={'type': 'song-choice', 'index': row['track_name']},
                        n_clicks=0
                    )
                )
            return html.Div([
                html.P(f"Found {len(exact_songs_df)} songs with same name, please try to fill with artist name for better recommendation."),
                *options
            ])

                # 3. Fuzzy match fallback
        all_songs = df["track_name"].dropna().tolist()
        closest_matches = difflib.get_close_matches(input_song_cleaned, all_songs, n=5, cutoff=0.75)
        if closest_matches:
            options = []
            for song in closest_matches:
                song_rows = df[df["track_name"].str.fullmatch(song, case=False, na=False)]
                for _, row in song_rows.iterrows():
                    options.append(
                        html.Button(
                            f"{row['track_name']} by {row['artist_name']}",
                            id={'type': 'song-choice', 'index': row['track_name']},
                            n_clicks=0
                        )
                    )
            return html.Div([
                html.P(f"Oops, we had issue to find exact match. Did you mean one of these {len(closest_matches)} songs? Try to search with it."),
                *options
            ])

        # If no matches at all
        return html.P("Darn! No matching songs found. Please try again with a different song.")

if __name__ == '__main__':
    app.run(debug=True)