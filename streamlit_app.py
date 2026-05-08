import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from pathlib import Path

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── OMDB API Key ──────────────────────────────────────────
import os
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

# ── Genre Background Mapping ──────────────────────────────
GENRE_BACKGROUNDS = {
    "Action":    "background/Action.jpg",
    "Adventure": "background/Adventure.png",
    "Animation": "background/Animation.jpg",
    "Children":  "background/Children.jpg",
    "Classic":   "background/Classic.jpg",
    "Comedy":    "background/Comedy.jpg",
    "Crime":     "background/Crime.jpg",
    "Drama":     "background/Drama.jpg",
    "Fantasy":   "background/Fantasy.jpg",
    "Film-Noir": "background/Film-noir.jpg",
    "Horror":    "background/Horror.jpg",
    "Musical":   "background/Musical.jpg",
    "Mystery":   "background/Mystery.jpg",
    "Romance":   "background/Romance.jpg",
    "Sci-Fi":    "background/Sci-fi.jpg",
    "Thriller":  "background/Thriller.png",
    "War":       "background/War.jpg",
    "Western":   "background/Western.jpg",
}

# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    movies_df  = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    item_sim   = np.load('item_similarity.npy')

    # Load pre-trained CF matrices
    # (no Surprise needed!)
    user_factors = np.load('user_factors.npy')
    item_factors = np.load('item_factors.npy')
    user_biases  = np.load('user_biases.npy')
    item_biases  = np.load('item_biases.npy')
    global_mean  = np.load('global_mean.npy')[0]
    item_map_df  = pd.read_csv('item_map.csv')
    item_map     = dict(zip(
        item_map_df['item_id'],
        item_map_df['idx']))

    # Add genres_str
    genre_cols = ['unknown','Action','Adventure',
                  'Animation','Children','Comedy',
                  'Crime','Documentary','Drama',
                  'Fantasy','Film-Noir','Horror',
                  'Musical','Mystery','Romance',
                  'Sci-Fi','Thriller','War','Western']
    existing = [c for c in genre_cols
                if c in movies_df.columns]
    if 'genres_str' not in movies_df.columns:
        movies_df['genres_str'] = movies_df[existing].apply(
            lambda row: ' '.join(
                [c for c in existing if row[c] == 1]),
            axis=1)

    cf_data = {
        'user_factors': user_factors,
        'item_factors': item_factors,
        'user_biases':  user_biases,
        'item_biases':  item_biases,
        'global_mean':  global_mean,
        'item_map':     item_map,
    }

    return cf_data, item_sim, movies_df, ratings_df

cf_data, item_similarity, movies, ratings = load_models()
# ── Base64 Image ──────────────────────────────────────────
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# ── Set Background ────────────────────────────────────────
def set_background(image_path):
    img_base64 = get_base64_image(image_path)
    if img_base64:
        ext = Path(image_path).suffix.replace(".", "")
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: #0f0f0f !important;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 700px;
            background-image: url(
                "data:image/{ext};base64,{img_base64}");
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: top center !important;
            background-color: #0f0f0f;
            z-index: 0;
        }}
        .block-container {{
            position: relative !important;
            z-index: 1 !important;
        }}
        </style>
        """, unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────
def apply_css():
    st.markdown("""
    <style>
    h1,h2,h3,h4,p,label,.stMarkdown {
        color: white !important;
        position: relative;
        z-index: 1;
    }
    .stTextInput > div > div > input {
        border-radius: 50px !important;
        border: 2px solid rgba(255,255,255,0.5) !important;
        background: rgba(0,0,0,0.5) !important;
        color: white !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }
    .stButton > button {
        border-radius: 50px !important;
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.4) !important;
        padding: 8px 16px !important;
        font-size: 13px !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        background: rgba(229,9,20,0.6) !important;
        border-color: #e50914 !important;
    }
    .movie-card {
        background: rgba(0,0,0,0.55);
        border-radius: 15px;
        padding: 15px;
        margin: 8px 0;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        color: white;
    }
    .rec-card {
        background: rgba(0,0,0,0.6);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(229,9,20,0.4);
        backdrop-filter: blur(10px);
    }
    .score-badge {
        text-align: center;
        padding: 20px;
        background: rgba(229,9,20,0.3);
        border-radius: 15px;
        border: 1px solid #e50914;
    }
    .no-poster {
        width: 120px;
        height: 180px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 40px;
        text-align: center;
    }
    .stRadio > div {
        flex-direction: row !important;
    }
    .stRadio label { color: white !important; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    .block-container {
        position: relative;
        z-index: 1;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(229,9,20,0.2) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ── OMDB API ──────────────────────────────────────────────
@st.cache_data
def get_movie_info(title):
    try:
        clean = title.split('(')[0].strip()
        url   = (f"http://www.omdbapi.com/"
                 f"?t={clean}&apikey={OMDB_API_KEY}")
        resp  = requests.get(url, timeout=5)
        d     = resp.json()
        if d.get('Response') == 'True':
            return {
                'poster':   d.get('Poster', ''),
                'plot':     d.get('Plot',
                            'No synopsis available.'),
                'year':     d.get('Year', 'N/A'),
                'imdb':     d.get('imdbRating', 'N/A'),
                'director': d.get('Director', 'N/A'),
                'actors':   d.get('Actors', 'N/A'),
                'genre':    d.get('Genre', 'N/A'),
                'runtime':  d.get('Runtime', 'N/A'),
                'awards':   d.get('Awards', 'N/A'),
                'language': d.get('Language', 'N/A'),
            }
    except:
        pass
    return {
        'poster':   '',
        'plot':     'Synopsis not available.',
        'year':     'N/A',
        'imdb':     'N/A',
        'director': 'N/A',
        'actors':   'N/A',
        'genre':    'N/A',
        'runtime':  'N/A',
        'awards':   'N/A',
        'language': 'N/A',
    }

# ── Poster Display ────────────────────────────────────────
def show_poster(poster_url, width=120):
    if poster_url and poster_url != 'N/A':
        st.image(poster_url, width=width)
    else:
        height = int(width * 1.5)
        st.markdown(f"""
        <div style='width:{width}px;height:{height}px;
        background:linear-gradient(135deg,
            rgba(229,9,20,0.3),rgba(0,0,0,0.5));
        border-radius:10px;display:flex;
        flex-direction:column;
        align-items:center;justify-content:center;
        color:white;border:1px solid
        rgba(255,255,255,0.2);'>
            <div style='font-size:30px;'>🎬</div>
            <div style='font-size:10px;
                        text-align:center;
                        padding:5px;
                        color:rgba(255,255,255,0.6);'>
                No Poster
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Hybrid Recommendations ────────────────────────────────
def get_hybrid_recommendations(user_ratings_input, n=10):
    liked_ids = [
        mid for mid, r in user_ratings_input.items()
        if r >= 4
    ]
    liked_indices = [
        movies[movies['item_id'] == i].index[0]
        for i in liked_ids
        if len(movies[movies['item_id'] == i].index) > 0
    ]

    rated_ids     = list(user_ratings_input.keys())
    all_movie_ids = movies['item_id'].tolist()
    unrated       = [m for m in all_movie_ids
                     if m not in rated_ids]

    # CF prediction using saved matrices
    def predict_cf(item_id):
        try:
            item_map = cf_data['item_map']
            if item_id not in item_map:
                return cf_data['global_mean']
            iid      = item_map[item_id]
            # Use average user factor
            u_factor = cf_data['user_factors'].mean(axis=0)
            i_factor = cf_data['item_factors'][iid]
            u_bias   = cf_data['user_biases'].mean()
            i_bias   = cf_data['item_biases'][iid]
            pred     = (cf_data['global_mean'] +
                       u_bias + i_bias +
                       np.dot(u_factor, i_factor))
            return float(np.clip(pred, 1, 5))
        except:
            return float(cf_data['global_mean'])

    scores = []
    for mid in unrated:
        # CF score
        cf_s    = predict_cf(mid)

        # CB score
        idx = movies[movies['item_id'] == mid].index
        if len(idx) > 0 and liked_indices:
            cb_s = float(
                item_similarity[
                    idx[0], liked_indices].mean())
        else:
            cb_s = 0.0

        cf_norm = (cf_s - 1) / 4
        h_score = 0.6 * cf_norm + 0.4 * cb_s
        scores.append((mid, h_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]

# ── Genre Movies ──────────────────────────────────────────
def get_genre_movies(genre_col, exclude_ids=[], n=20):
    if genre_col in movies.columns:
        gm = movies[movies[genre_col] == 1].copy()
    else:
        gm = movies.copy()

    avg_ratings      = ratings.groupby(
        'item_id')['rating'].mean()
    gm['avg_rating'] = gm['item_id'].map(avg_ratings)
    gm               = gm.dropna(subset=['avg_rating'])
    gm               = gm[~gm['item_id'].isin(exclude_ids)]
    gm               = gm.sort_values(
        'avg_rating', ascending=False)
    return gm.head(n)

# ── Session State ─────────────────────────────────────────
for key, val in {
    'selected_genre': None,
    'user_ratings':   {},
    'refresh_count':  0,
    'show_recs':      False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Background ────────────────────────────────────────────
if (st.session_state.selected_genre and
        st.session_state.selected_genre
        in GENRE_BACKGROUNDS):
    set_background(
        GENRE_BACKGROUNDS[st.session_state.selected_genre])
else:
    st.markdown("""
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: linear-gradient(
            135deg, #0f0f0f 0%,
            #1a1a2e 50%, #16213e 100%) !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

apply_css()

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:20px 0;'>
    <h1 style='font-size:48px;font-weight:900;
               color:white;letter-spacing:2px;'>
        🎬 Movie Recommender
    </h1>
    <p style='color:rgba(255,255,255,0.7);font-size:16px;'>
        🎥 Movies from 1922 to 1998 •
        Search, Rate &amp; Discover!
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# SEARCH BAR
# ══════════════════════════════════════════════════════════
st.markdown(
    "<h3 style='color:white;'>🔍 Search a Movie</h3>",
    unsafe_allow_html=True)

# Dataset year note
st.markdown("""
<p style='color:rgba(255,255,255,0.5);font-size:12px;
          margin-top:-10px;'>
    ℹ️ This app uses MovieLens 100K dataset
    (movies from 1922–1998 only)
</p>
""", unsafe_allow_html=True)

search_query = st.text_input(
    "search",
    placeholder="🔍  Search for a movie (e.g. Titanic, "
                "Toy Story, Casablanca)...",
    label_visibility="collapsed"
)

# ── Search Results ────────────────────────────────────────
if search_query and len(search_query) >= 2:
    matched = movies[
        movies['title'].str.contains(
            search_query, case=False, na=False)
    ].head(5)

    if not matched.empty:
        st.markdown(
            f"<h4 style='color:white;'>"
            f"Found {len(matched)} result(s):</h4>",
            unsafe_allow_html=True)

        for _, row in matched.iterrows():
            movie_id = int(row['item_id'])
            title    = row['title']
            info     = get_movie_info(title)

            with st.container():
                st.markdown(
                    "<div class='movie-card'>",
                    unsafe_allow_html=True)

                col1, col2 = st.columns([1, 3])
                with col1:
                    show_poster(info['poster'], width=120)

                with col2:
                    st.markdown(f"""
                    <h4 style='color:white;margin:0;'>
                        {title}
                    </h4>
                    <p style='color:rgba(255,255,255,0.6);
                              font-size:12px;margin:4px 0;'>
                        📅 {info['year']} &nbsp;|&nbsp;
                        ⭐ IMDB: {info['imdb']}
                        &nbsp;|&nbsp;
                        ⏱️ {info['runtime']}
                        &nbsp;|&nbsp;
                        🌍 {info['language']}
                    </p>
                    <p style='color:rgba(255,255,255,0.85);
                              font-size:13px;
                              margin:6px 0;'>
                        📖 {info['plot']}
                    </p>
                    <p style='color:rgba(255,255,255,0.6);
                              font-size:12px;'>
                        🎬 <b>Director:</b>
                        {info['director']}<br>
                        👤 <b>Cast:</b>
                        {info['actors']}<br>
                        🏆 <b>Awards:</b>
                        {info['awards']}
                    </p>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        "<p style='color:white;"
                        "margin:6px 0;font-weight:bold;'>"
                        "⭐ Rate this movie:</p>",
                        unsafe_allow_html=True)

                    rating_val = st.radio(
                        f"rate_{movie_id}",
                        options=[1, 2, 3, 4, 5],
                        horizontal=True,
                        key=f"search_rating_{movie_id}",
                        label_visibility="collapsed"
                    )
                    if st.button(
                        "✅ Submit Rating",
                        key=f"search_submit_{movie_id}"
                    ):
                        st.session_state.user_ratings[
                            movie_id] = rating_val
                        st.session_state.show_recs = True
                        st.success(
                            f"✅ Rated '{title}' "
                            f"{'⭐' * rating_val}!")
                        st.rerun()

                st.markdown(
                    "</div>", unsafe_allow_html=True)
    else:
        st.warning(
            f"No movies found for '{search_query}'. "
            f"Remember: only movies from 1922–1998!")

# ══════════════════════════════════════════════════════════
# GENRE SECTION
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<h3 style='color:white;'>🎭 Browse by Genre</h3>",
    unsafe_allow_html=True)

genres = [
    "Action",   "Adventure", "Animation", "Children",
    "Classic",  "Comedy",    "Crime",     "Drama",
    "Fantasy",  "Film-Noir", "Horror",    "Musical",
    "Mystery",  "Romance",   "Sci-Fi",    "Thriller",
    "War",      "Western"
]

cols = st.columns(6)
for i, genre in enumerate(genres):
    with cols[i % 6]:
        is_active = st.session_state.selected_genre == genre
        label     = f"✅ {genre}" if is_active else genre
        if st.button(label, key=f"genre_btn_{genre}"):
            if st.session_state.selected_genre == genre:
                st.session_state.selected_genre = None
            else:
                st.session_state.selected_genre = genre
                st.session_state.refresh_count  = 0
            st.rerun()

# ══════════════════════════════════════════════════════════
# GENRE MOVIES
# ══════════════════════════════════════════════════════════
if st.session_state.selected_genre:
    genre = st.session_state.selected_genre

    st.markdown(f"""
    <h3 style='color:white;margin-top:20px;'>
        🎬 Top {genre} Movies
    </h3>
    """, unsafe_allow_html=True)

    genre_col_map = {
        "Film-Noir": "Film-Noir",
        "Sci-Fi":    "Sci-Fi",
        "Classic":   "Drama",
        "Children":  "Children",
    }
    col_name = genre_col_map.get(genre, genre)

    offset           = st.session_state.refresh_count * 10
    all_genre_movies = get_genre_movies(
        col_name,
        exclude_ids=list(
            st.session_state.user_ratings.keys()),
        n=offset + 10
    )
    page_movies = all_genre_movies.iloc[offset:offset + 10]

    if page_movies.empty:
        st.warning("No more movies! Click Refresh.")
        st.session_state.refresh_count = 0
    else:
        r1, r2 = st.columns([5, 1])
        with r2:
            if st.button("🔄 Refresh",
                         key="refresh_btn"):
                st.session_state.refresh_count += 1
                st.rerun()

        movie_list = page_movies.to_dict('records')
        for i in range(0, len(movie_list), 2):
            gcols = st.columns(2)
            for j, gcol in enumerate(gcols):
                if i + j < len(movie_list):
                    mv    = movie_list[i + j]
                    title = mv['title']
                    mid   = int(mv['item_id'])
                    info  = get_movie_info(title)

                    with gcol:
                        st.markdown(
                            "<div class='movie-card'>",
                            unsafe_allow_html=True)
                        c1, c2 = st.columns([1, 2])

                        with c1:
                            show_poster(
                                info['poster'], width=90)

                        with c2:
                            st.markdown(f"""
                            <p style='color:white;
                            font-weight:bold;
                            font-size:13px;margin:0;'>
                                {title}
                            </p>
                            <p style='color:rgba
                            (255,255,255,0.6);
                            font-size:11px;margin:2px 0;'>
                                📅 {info['year']} |
                                ⭐ {info['imdb']} |
                                ⏱️ {info['runtime']}
                            </p>
                            """, unsafe_allow_html=True)

                            # Expandable synopsis
                            with st.expander(
                                    "📖 Synopsis & Details"):
                                st.markdown(f"""
                                <p style='color:white;
                                font-size:12px;'>
                                    <b>Plot:</b>
                                    {info['plot']}<br><br>
                                    <b>🎬 Director:</b>
                                    {info['director']}<br>
                                    <b>👤 Cast:</b>
                                    {info['actors']}<br>
                                    <b>🏆 Awards:</b>
                                    {info['awards']}<br>
                                    <b>🌍 Language:</b>
                                    {info['language']}
                                </p>
                                """,
                                unsafe_allow_html=True)

                            r_val = st.radio(
                                "Rate:",
                                [1, 2, 3, 4, 5],
                                horizontal=True,
                                key=f"genre_r_{mid}",
                                label_visibility="collapsed"
                            )
                            if st.button(
                                "⭐ Rate",
                                key=f"genre_b_{mid}"
                            ):
                                st.session_state\
                                    .user_ratings[
                                    mid] = r_val
                                st.session_state\
                                    .show_recs = True
                                st.success(
                                    f"Rated! {'⭐'*r_val}")
                                st.rerun()

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
if st.session_state.show_recs and \
        st.session_state.user_ratings:
    st.markdown("---")
    st.markdown("""
    <h2 style='color:white;text-align:center;'>
        🎯 Your Personalised Recommendations
    </h2>
    <p style='color:rgba(255,255,255,0.7);
              text-align:center;font-size:15px;'>
        Powered by Hybrid AI
        (Collaborative + Content-Based Filtering)
    </p>
    """, unsafe_allow_html=True)

    rated_count = len(st.session_state.user_ratings)
    st.markdown(
        f"<p style='color:rgba(255,255,255,0.8);'>"
        f"🎬 Based on your "
        f"{rated_count} rating(s):</p>",
        unsafe_allow_html=True)

    for mid, r in st.session_state.user_ratings.items():
        t = movies[movies['item_id'] == mid]['title']
        if not t.empty:
            st.markdown(
                f"<p style='color:rgba(255,255,255,0.6);"
                f"font-size:12px;margin:2px;'>"
                f"  {'⭐'*r} {t.values[0]}</p>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner(
            "🎯 Finding best movies for you..."):
        recs = get_hybrid_recommendations(
            st.session_state.user_ratings, n=10)

    st.markdown(
        "<h3 style='color:white;'>🎬 Top 10 Picks "
        "For You:</h3>",
        unsafe_allow_html=True)

    for rank, (movie_id, score) in enumerate(recs, 1):
        row = movies[movies['item_id'] == movie_id]
        if row.empty:
            continue

        title      = row['title'].values[0]
        genres_str = row['genres_str'].values[0] \
            if 'genres_str' in row.columns else ""
        info       = get_movie_info(title)

        st.markdown(
            "<div class='rec-card'>",
            unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 3, 1])

        with c1:
            show_poster(info['poster'], width=120)

        with c2:
            st.markdown(f"""
            <h4 style='color:white;margin:0;'>
                #{rank}. {title}
            </h4>
            <p style='color:rgba(255,255,255,0.6);
                      font-size:12px;margin:4px 0;'>
                📅 {info['year']} &nbsp;|&nbsp;
                ⭐ IMDB: {info['imdb']}
                &nbsp;|&nbsp;
                ⏱️ {info['runtime']}
            </p>
            <p style='color:rgba(255,255,255,0.7);
                      font-size:12px;margin:2px 0;'>
                🎭 {genres_str}
            </p>
            """, unsafe_allow_html=True)

            # ── Expandable full details ───────────────
            with st.expander("📖 Full Synopsis & Details"):
                dc1, dc2 = st.columns([1, 2])
                with dc1:
                    show_poster(info['poster'], width=150)
                with dc2:
                    st.markdown(f"""
                    <p style='color:white;font-size:13px;'>
                        <b>📖 Plot:</b><br>
                        {info['plot']}<br><br>
                        <b>🎬 Director:</b>
                        {info['director']}<br>
                        <b>👤 Cast:</b>
                        {info['actors']}<br>
                        <b>🎭 Genre:</b>
                        {info['genre']}<br>
                        <b>🏆 Awards:</b>
                        {info['awards']}<br>
                        <b>🌍 Language:</b>
                        {info['language']}
                    </p>
                    """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='score-badge'>
                <p style='color:white;
                          font-size:12px;margin:0;'>
                    Match Score
                </p>
                <h2 style='color:#e50914;margin:0;'>
                    {score:.0%}
                </h2>
                <p style='color:rgba(255,255,255,0.6);
                          font-size:10px;margin:4px 0;'>
                    Hybrid AI
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            "</div><br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start Over", key="start_over"):
        st.session_state.user_ratings   = {}
        st.session_state.show_recs      = False
        st.session_state.selected_genre = None
        st.session_state.refresh_count  = 0
        st.rerun()

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:30px;
            color:rgba(255,255,255,0.4);
            font-size:12px;'>
    Built with ❤️ using MovieLens 100K Dataset
    (1922–1998) •
    SVD + TF-IDF + Hybrid Model
</div>
""", unsafe_allow_html=True)