# What's Your Jam Final Project
---------------------------------------------

**Overview**
We decided to create a music recommender from a machine learning model trained on a data set that was publicly avaialable on kaggle, and pulled from the Spotify API. The idea behind this music recomender in particular is to recommend songs that may not be in the same genre, but share some of the same characteristics as one another. To do this, we built a KMeans machine learning model to create clusters of songs that are grouped together based on different features. This allows us to have an output of similar songs to any given input, so long as the chosen song is in the dataset. To make this an interactive experience for the user, we opted to use the Dash library for creating a Dashboard. The dataset is very comprehensive, containing a little over 232,000 songs, as well as many features about the songs. The songs features allow the model to more accurately predict how the songs are related to each other past just genre. The features are as follows: popularity, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence.

The definitions for the features per the Spotify for Developers website:

Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high
confidence the track is acoustic.

Danceability: It describes how suitable a track is for dancing based on a combination of musical
elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least
danceable and 1.0 is most danceable.

Duration_ms: The duration of the track in milliseconds

Energy: This is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a
Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic
range, perceived loudness, timbre, onset rate, and general entropy.

Instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as
instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the
instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above
0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

Key: the key the track is in. Integers map to pitches using standard Pitch Class notation. If no key was detected, the value is -1.

Speechiness: It detects the presence of spoken words in a track. The more exclusively speech-like the
recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66
describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe
tracks that may contain both music and speech, either in sections or layered, including such cases as rap
music. Values below 0.33 most likely represent music and other non-speech-like tracks.

Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

Time_signature: an estimated time signature. The time signature(meter) is a notation convention to specify how many beats are in each bar (or measure).

Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).


Forrest Margulies, Gurpreet Singh Badrani, Luis Lopez, Yue Deng