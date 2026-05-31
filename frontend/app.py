"""
app.py
------
Streamlit frontend for the movie recommender.

Runs on Hugging Face Spaces (streamlit SDK) or locally:
    streamlit run frontend/app.py

Set the API URL via env var or the sidebar input:
    export API_URL=https://your-service.onrender.com
"""

import os
import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Filmfinder — Neural Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS  ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&display=swap');
  h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
  .stButton > button {
      border-radius: 2px !important;
      font-family: monospace !important;
      letter-spacing: 0.1em !important;
      text-transform: uppercase !important;
  }
  .movie-card {
      border: 1.5px solid #d4c9b4;
      border-radius: 3px;
      padding: 14px;
      background: #ffffff;
      margin-bottom: 10px;
  }
  .score-bar { background: #e8ddd0; border-radius: 2px; height: 4px; margin-top: 6px; }
  .score-fill { background: #d4892a; height: 4px; border-radius: 2px; }
  .genre-tag {
      display: inline-block;
      border: 1px solid #d4c9b4;
      padding: 2px 7px; border-radius: 2px;
      font-size: 11px; color: #8a7d6b; margin-right: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Filmfinder")
    st.caption("Two-Tower Neural Recommender · MovieLens 100K")

    api_url = st.text_input(
        "API URL",
        value=os.environ.get("API_URL", "http://localhost:8000"),
        help="Your FastAPI server URL (Render, HF Spaces, or localhost)",
    )

    # Health check
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.ok and r.json().get("status") == "ok":
            info = r.json()
            st.success(f"✓ Live — {info['n_users']} users, {info['n_items']} items")
        else:
            st.warning("API reachable but not ready")
    except Exception:
        st.error("⚠ API offline — check your URL")

    st.divider()
    mode = st.radio("Mode", ["Known User", "Cold Start"])
    top_k = st.slider("Top-K results", 5, 20, 10)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# Your *picks*")

def render_movie(i: int, m: dict):
    genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in m.get("genres", [])[:4])
    score_pct   = int(m.get("score", 0) * 100)
    st.markdown(f"""
    <div class="movie-card">
      <small style="color:#8a7d6b;letter-spacing:0.15em;text-transform:uppercase">#{i+1}</small>
      <h3 style="margin:4px 0 2px">{m['title']}</h3>
      <div style="font-size:12px;color:#8a7d6b;margin-bottom:8px">{m.get('year','')}</div>
      <div>{genres_html}</div>
      <div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>
      <small style="color:#8a7d6b">similarity: {m.get('score', 0):.3f}</small>
    </div>
    """, unsafe_allow_html=True)


if mode == "Known User":
    user_id = st.number_input("User ID (1–943)", min_value=1, max_value=943, value=42)
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Encoding vectors…"):
            try:
                r = requests.post(
                    f"{api_url}/recommend",
                    json={"user_id": user_id, "top_k": top_k, "exclude_seen": True},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                st.caption(f"mode: {data['mode']} · model: {data['model_version']}")
                cols = st.columns(3)
                for i, movie in enumerate(data["results"]):
                    with cols[i % 3]:
                        render_movie(i, movie)
            except Exception as e:
                st.error(f"Request failed: {e}")

else:  # Cold Start
    st.info("Enter comma-separated movie IDs you like (e.g. 50, 172, 302)")
    raw = st.text_input("Liked movie IDs", "50, 172, 302")
    if st.button("Get Recommendations", type="primary"):
        try:
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            st.error("Please enter valid integer movie IDs separated by commas.")
            ids = []
        if ids:
            with st.spinner("Encoding vectors…"):
                try:
                    r = requests.post(
                        f"{api_url}/recommend",
                        json={"liked_movie_ids": ids, "top_k": top_k},
                        timeout=10,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.caption(f"mode: {data['mode']} · model: {data['model_version']}")
                    cols = st.columns(3)
                    for i, movie in enumerate(data["results"]):
                        with cols[i % 3]:
                            render_movie(i, movie)
                except Exception as e:
                    st.error(f"Request failed: {e}")