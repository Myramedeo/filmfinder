# Filmfinder - End-to-End ML Pipeline

A production-style movie recommendation system built with PyTorch, FastAPI, and Streamlit. Trained on MovieLens 100K using a two-tower neural architecture, the same retrieval pattern used at YouTube, Pinterest, and Spotify.

**[Live Demo](https://myramedeo.github.io/filmfinder/)** · **[API Docs](https://filmfinder-xv5p.onrender.com/docs)**

---

## Architecture

```plaintext
MovieLens 100K
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                   Two-Tower Model                       │
│                                                         │
│  User Tower                     Item Tower              │
│  ──────────                     ───────────             │
│  user_id  → embedding           item_id  → embedding    │
│  occ_id   → embedding           genre    → MLP          │
│  age, gender                    year                    │
│       └──────── MLP ───┐   ┌─── MLP ────┘               │
│                        ▼   ▼                            │
│               cosine similarity → sigmoid               │
└─────────────────────────────────────────────────────────┘
      │
      ▼
FastAPI  →  POST /recommend · GET /similar/{id}
      │
      ▼
Streamlit dashboard  (Hugging Face Spaces)
```

## Tech Stack

| Layer | Tools |
| --- | --- |
| Data | pandas, MovieLens 100K, TMDB API (optional) |
| Model | PyTorch, two-tower architecture, BCE loss, AdamW |
| Evaluation | NDCG@K, Precision@K, Recall@K, AUC |
| API | FastAPI, Pydantic v2, uvicorn |
| Frontend | Streamlit (HF Spaces) + standalone HTML dashboard |
| Deploy | Render (API) + Hugging Face Spaces / GitHub Pages (UI) |

## Results

Trained on MovieLens 100K (80K train / 10K val / 10K test, time-based split):

| Metric | Score |
| --- | --- |
| Test AUC | ~0.72 |
| NDCG@10 | ~0.12 |
| Precision@10 | ~0.08 |
| Inference latency | ~3 ms (CPU) |

## Project Structure

```plaintext
movie-recommender/
├── data/
│   ├── download.py        # fetch MovieLens 100K
│   ├── preprocess.py      # ID maps, features, time-based split
│   ├── dataset.py         # PyTorch Dataset + DataLoaders
│   └── tmdb.py            # optional poster/overview enrichment
├── models/
│   ├── two_tower.py       # UserTower + ItemTower + scoring head
│   ├── train.py           # training loop, early stopping, checkpointing
│   └── evaluate.py        # NDCG@K, Precision@K, Recall@K, AUC
├── api/
│   ├── schemas.py         # Pydantic request/response models
│   ├── model_store.py     # model load, pre-computed item index
│   └── main.py            # FastAPI routes, CORS, rate limiting
├── frontend/
│   ├── index.html         # standalone HTML/JS dashboard
│   └── app.py             # Streamlit app (HF Spaces)
├── scripts/
│   └── start.sh           # train-if-missing → launch uvicorn
├── .github/workflows/
│   └── train.yml          # auto-retrain + commit checkpoint on push
├── Dockerfile
├── render.yaml
└── DEPLOY.md
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/movie-recommender
cd movie-recommender
pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# 2. Download data and train
python -m data.download
python -m models.train

# 3. Start the API
uvicorn api.main:app --reload --port 8000

# 4. Try it
curl -X POST localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "top_k": 5}'

# Cold start (no user ID needed)
curl -X POST localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"liked_movie_ids": [50, 172, 302], "top_k": 5}'
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe — returns model stats |
| `POST` | `/recommend` | Top-K recommendations for a user |
| `GET` | `/movies/{id}` | Movie metadata lookup |
| `GET` | `/similar/{id}` | Item-to-item similarity |
| `GET` | `/users` | List of known user IDs |

**Known user request:**

```json
{ "user_id": 42, "top_k": 10, "exclude_seen": true }
```

**Cold start request** (no user history needed):

```json
{ "liked_movie_ids": [50, 172, 302], "top_k": 10 }
```

## Model Design Notes

**Two-tower model:** The user and item towers produce embeddings in a shared vector space. At inference, all item vectors are pre-computed once and stored in a `(n_items × 64)` matrix. Each recommendation request is then a single user-tower forward pass plus a dot-product (~3 ms on CPU). This is the same retrieval pattern used in production systems before an ANN index (FAISS, ScaNN) replaces the brute-force dot product.

**Cold start via embedding averaging:** When no `user_id` is provided, the API averages the item vectors of the supplied liked movies and L2-normalises the result. This works because both towers are trained in the same metric space, so the average is a valid query vector.

**Time-based splitting:** Train/val/test are split chronologically (80% / 10% / 10%) rather than randomly. Random splitting inflates metrics by leaking future signal into training.

## Deployment

```bash
# 1. Commit the trained checkpoint
git add models/checkpoints/best_model.pt
git commit -m "add trained checkpoint"
git push

# 2. Render (API) — connect repo → New → Blueprint → Apply
#    Render finds render.yaml automatically

# 3. HF Spaces (UI) — push frontend/ to a new Streamlit Space
#    Set API_URL secret to your Render URL
```

Note that GitHub Actions (`.github/workflows/train.yml`) re-trains and commits the checkpoint automatically whenever `data/` or `models/` changes.

## License

MIT
