---
title: Filmfinder — Neural Movie Recommender
emoji: 🎬
colorFrom: yellow
colorTo: orange
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# Filmfinder — Two-Tower Neural Movie Recommender

A full end-to-end ML pipeline: data ingestion → model training → live API → this dashboard.

## Architecture

```
MovieLens 100K  →  Two-Tower PyTorch Model  →  FastAPI (Render)  →  This UI
```

- **User tower**: user embedding + occupation embedding + age + gender → 64-d vector  
- **Item tower**: item embedding + genre MLP + release year → 64-d vector  
- **Scoring**: cosine similarity with learnable temperature → sigmoid  
- **Retrieval**: pre-computed item matrix, dot-product, top-K in ~3 ms  

## Modes

| Mode | How it works |
|---|---|
| **Known user** | Looks up the trained user embedding by ID (943 users) |
| **Cold start** | Averages item vectors of liked movies → approximate user vector |

## Tech stack

`Python` · `PyTorch` · `FastAPI` · `Streamlit` · `Render` · `Hugging Face Spaces`

## Source code

→ [GitHub repo](https://github.com/Myramedeo/filmfinder)