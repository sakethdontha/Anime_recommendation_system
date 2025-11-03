# Anime Recommendation System (Text + Image)

This repository contains a content‑based anime recommender that blends **text similarity** (synopsis + metadata) with **image similarity** (poster thumbnails) using a weighted fusion:

- **Text features:** `CountVectorizer` over a `tags` field built from *synopsis*, *rating*, *genres*, *type*, and *status* → cosine similarity.
- **Image features:** **CLIP** (`openai/clip-vit-base-patch32`) embeddings for anime poster images → cosine similarity.
- **Final score:** `0.7 * text_similarity + 0.3 * image_similarity`

The notebook produces two pickled artifacts for fast serving:
- `artifacts/anime_list.pkl` — the final DataFrame with essentials (`id`, `name`, `tags`, `image`).
- `artifacts/similarity.pkl` — the fused similarity matrix aligned to `anime_list.pkl` rows.

A simple `recommend(name)` helper demonstrates how to fetch similar titles by name.

---

## Project Structure

```
.
├── recommendation_sys.ipynb        # Main notebook (EDA → feature build → CLIP → fusion → artifacts)
├── popular_anime.csv               # Input CSV with anime metadata (see required columns below)
└── artifacts/
    ├── anime_list.pkl              # Saved DataFrame used by apps
    └── similarity.pkl              # N×N similarity matrix (cosine-based) for recommendations
```

> **Tip:** The notebook also saves `clip_image_features_batched.npy` during feature extraction for reuse/debugging.

---

## Data Requirements

The notebook expects a CSV named **`popular_anime.csv`** with the following columns (minimum):

- `id` — unique identifier
- `name` — anime title (must be unique enough to index by name)
- `genres` — pipe/comma‑separated genres
- `type` — TV/Movie/OVA/etc.
- `status` — e.g., Finished/Airing
- `aired_from` — date or year (optional for modeling, used in EDA)
- `duration_per_ep` — human‑readable duration (e.g., `24 min`, `1 hr 30 min`)
- `score`, `scored_by`, `rating`, `studios` — optional but kept in the final table
- `synopsis` — main text used for text features
- `image` — URL to poster/cover image (used for CLIP)

> The notebook includes a `parsing_duration()` utility to standardize `duration_per_ep` (minutes).

---

## How It Works

1. **Load & Select Columns**
   - Read `popular_anime.csv` into `df` and subset to the working columns.
2. **Minimal Cleaning & Tag Build**
   - Create `anime['tags'] = synopsis + rating + genres + type + status`.
   - Vectorize with `CountVectorizer(max_features=8000, stop_words="english")` → `vector`.
3. **Text Similarity**
   - `similarity = cosine_similarity(vector)`
4. **Image Similarity (CLIP)**
   - Download images from `anime['image']` (robust to failures; falls back to zero vectors).
   - Encode with `CLIPProcessor/CLIPModel` (`openai/clip-vit-base-patch32`) on CPU/GPU.
   - `image_sim = cosine_similarity(clip_features)`
5. **Fusion**
   - `final_sim = 0.7 * similarity + 0.3 * image_sim`
6. **Artifacts**
   - `final = anime[['id','name','tags','image']]`
   - Save `final` and `final_sim` to `artifacts/` as pickles.

---

## Quickstart (Local)

### 1) Environment

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

**If you don’t have a requirements file yet, these are the core libs:**
```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
requests
Pillow
transformers
torch          # CPU or CUDA build
sentence-transformers
nltk
chromadb       # present in notebook; not mandatory for the basic flow
```

> For GPU: install a CUDA‑compatible `torch` build from https://pytorch.org

### 2) Files

- Place **`popular_anime.csv`** at the repo root.
- Create the `artifacts/` folder if it doesn’t exist.

### 3) Run the Notebook

Open `recommendation_sys.ipynb` and run all cells. This will:
- Download CLIP weights on first run.
- Compute text & image similarities.
- Save `artifacts/anime_list.pkl` and `artifacts/similarity.pkl`.

> First run can take time due to image downloads + CLIP inference.

---

Wrapped this in **Streamlit** to serve a simple UI that lists similar titles and shows poster images.

---

## Reproducibility Notes

- CLIP image similarities depend on image download success; the notebook includes a fallback (`np.zeros(512)`) for failed images.
- If your CSV has duplicate names, de‑duplicate or index by `id` and map names → ids in your app.
- Text similarity is driven by the `tags` field. You can improve results by:
  - Adding more fields (e.g., `studios`, `scored_by`, user tags)
  - Switching to `TfidfVectorizer`
  - Doing more aggressive text cleaning (stopwords, stemming/lemmatization)
- Weighting (`0.7/0.3`) is a simple heuristic—tune for your dataset.

---

## Troubleshooting

- **CLIP is slow on CPU** → Try a CUDA build of `torch` and ensure `torch.cuda.is_available()` is `True`.
- **Image downloads fail** → Check `image` URLs and your network. You can pre‑download images or cache them locally.
- **Artifacts not created** → Ensure `artifacts/` exists and the notebook runs to completion.
- **KeyError on name** → The title passed to `recommend()` must exactly match `anime_list['name']`. Consider implementing fuzzy search.

---

## License

This project is for educational purposes. If you plan to distribute a public service or trained artifacts, review the licenses for the dataset and for **Hugging Face Transformers** and **OpenAI CLIP** weights.

---

## Acknowledgments

- **CLIP**: Radford et al., 2021, implemented via 🤗 **Transformers**.
- Scikit‑learn for vectorization and cosine similarity.
- The anime metadata CSV is expected to come from your own curation or a public dataset that permits such use.
