from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, UploadFile, File
from fastapi import Depends, FastAPI
from fastapi_cache import caches, close_caches
from cachetools import cached, TTLCache
from fastapi_cache.backends.redis import CACHE_KEY, RedisCacheBackend
from PIL import Image
from sentence_transformers import util
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from googletrans import Translator
import requests
import pickle
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
bearer_scheme = HTTPBearer()
cache = TTLCache(maxsize=1000, ttl=300)  # Cache sử dụng TTL (time-to-live) là 300 giây (5 phút)
translator = Translator()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

emb_path = "app/wallpaper_indexing/wallpaper-embeddings.pkl"


with open(emb_path, "rb") as fIn:
    img_names, img_emb = pickle.load(fIn)

def redis_cache():
    return caches.get(CACHE_KEY)

async def startup():
    rc = RedisCacheBackend("redis://redis")
    caches.set(CACHE_KEY, rc)

async def shutdown():
    await close_caches()

@app.on_event("startup")
async def app_startup():
    await startup()

@app.on_event("shutdown")
async def app_shutdown():
    await shutdown()

users = [
    {"username": "admin", "password": "ecowallpaper"},
    {"username": "user1", "password": "ecowallpaper1"},
    {"username": "user2", "password": "ecowallpaper2"},
]

def authenticate(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        token = credentials.credentials  # Lấy token từ header Authorization

        for user in users:
            if token == user["password"]:
                return {"username": user["username"]}

        raise Exception("Invalid token")
    except Exception as e:

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(authenticate)])
def homepage():
    """
    Endpoint for homepage
    """
    return """
    <pre>
    <h1>Welcome to the Recommend Wallpaper V2 REST API!</h1>
    <p>This API provides endpoints for searching and recommending wallpapers.</p>
    <p>Available Endpoints:</p>
    <ul>
        <li>/search?text={search_text}&page={page}&per_page={per_page} [GET] - Search wallpapers by text</li>
        <li>/search/image?page={page}&per_page={per_page} [POST] - Search wallpapers by image</li>
        <li>/recommend?thumbnail_storage_file_name={thumbnail_storage_file_name}&page={page}&per_page={per_page} [GET] - Recommend wallpapers based on an image</li>
        <li>/add_wallpaper?thumbnail_storage_file_name={thumbnail_storage_file_name} [GET] - Add a new wallpaper</li>
    </ul>
    </pre>
    """



@app.get("/add_wallpaper", dependencies=[Depends(authenticate)])
async def add_new_wallpaper(thumbnail_storage_file_name: str):
    global img_names, img_emb
    data = {"success": False}

    if thumbnail_storage_file_name in img_names:
        data["message"] = "wallpaper already exists!"
        return data
    try:
        image = Image.open(requests.get(f"https://wallpapernew.net/api/static/images/storage/{thumbnail_storage_file_name}",
                                        stream=True).raw).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        data["message"] = "Error loading image!"
        return data

    images = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to("cpu")

    query_emb = model.get_image_features(images)
    batch_emb = query_emb.squeeze(0).cpu().detach().numpy()

    img_names.append(thumbnail_storage_file_name)
    img_emb_new = np.vstack((img_emb, batch_emb))

    with open(emb_path, "wb") as fOut:
        pickle.dump((img_names, img_emb_new), fOut)

    with open(emb_path, "rb") as fIn:
        img_names, img_emb = pickle.load(fIn)


    data["success"] = True
    data["message"] = "Wallpaper added successfully"

    return data

@app.get("/search", dependencies=[Depends(authenticate)])
@cached(cache)
def search_text(text: str, page: int = 1, per_page: int = 10):
    data = {"success": False}

    translated_query = translator.translate(text=text, dest='en').text

    inputs = tokenizer([translated_query], padding=True, return_tensors="pt")
    query_emb = model.get_text_features(**inputs)

    all_hits = util.semantic_search(query_emb, img_emb, top_k=per_page * page)[0]

    start_idx = (page - 1) * per_page
    end_idx = page * per_page
    hits = all_hits[start_idx:end_idx]

    images = [
        {
            "thumbnail_storage_file_name": img_names[hit["corpus_id"]],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{img_names[hit['corpus_id']]}",
            "score": f"{hit['score'] * 100:.2f}%",
        }
        for hit in hits
    ]
    data["success"] = True
    data["text_search"] = text
    data["data"] = images

    return data


@app.post("/search/image", dependencies=[Depends(authenticate)])
def search_image(page: int = 1, per_page: int = 10, image: UploadFile = File(...)):
    data = {"success": False}
    image = Image.open(image.file)

    images = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to("cpu")

    query_emb = model.get_image_features(images)

    all_hits = util.semantic_search(query_emb, img_emb, top_k=per_page * page)[0]

    start_idx = (page - 1) * per_page
    end_idx = page * per_page
    hits = all_hits[start_idx:end_idx]

    images = [
        {
            "image_name": img_names[hit["corpus_id"]],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{img_names[hit['corpus_id']]}",
            "score": f"{hit['score'] * 100:.2f}%",
            "id": hit["corpus_id"]
        }
        for hit in hits
    ]

    data["success"] = True
    data["data"] = images

    return data


@app.get("/recommend", dependencies=[Depends(authenticate)])
def recommend_images(thumbnail_storage_file_name: str, page: int = 1, per_page: int = 10):
    global img_names, img_emb
    data = {"success": False}

    if thumbnail_storage_file_name in img_names:
        idx = img_names.index(thumbnail_storage_file_name)
        query_emb = img_emb[idx]
    else:
        data["message"] = "Image not learned!"
        try:
            image = Image.open(requests.get(f"https://wallpapernew.net/api/static/images/storage/{thumbnail_storage_file_name}",
                                            stream=True).raw).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            data["message"] = "Error loading image!"
            return data

        images = processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to("cpu")

        query_emb = model.get_image_features(images)
        batch_emb = query_emb.squeeze(0).cpu().detach().numpy()

        img_names.append(thumbnail_storage_file_name)
        img_emb_new = np.vstack((img_emb, batch_emb))

        with open(emb_path, "wb") as fOut:
            pickle.dump((img_names, img_emb_new), fOut)

        with open(emb_path, "rb") as fIn:
            img_names, img_emb = pickle.load(fIn)

        idx = img_names.index(thumbnail_storage_file_name)
        query_emb = img_emb[idx]

    all_hits = util.semantic_search(query_emb, img_emb, top_k=per_page * page)[0]

    start_idx = (page - 1) * per_page
    end_idx = page * per_page
    hits = all_hits[start_idx:end_idx]

    images = [
        {
            "image_name": img_names[hit["corpus_id"]],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{img_names[hit['corpus_id']]}",
            "score": f"{hit['score'] * 100:.2f}%",
            "id": hit["corpus_id"]
        }
        for hit in hits
    ]

    data["success"] = True
    data["data"] = images
    return data
