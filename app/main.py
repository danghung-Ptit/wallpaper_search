from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, UploadFile, File, Form, Query
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
import torch
from typing import List, Tuple, Optional, Union

app = FastAPI()
bearer_scheme = HTTPBearer()
cache = TTLCache(maxsize=10000, ttl=3600)
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
    {"username": "admin", "password": "849e0bb303d44818bcf8cea4f4a9fd6cfb26a1ee3b8cbf7e73f1500a666f66d29cc4560c020b20460889b0a9c236b49e1ebf14910b858567278f44eff16eba03"},
    {"username": "user2", "password": "ecowallpaper111"},
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
    </pre>
    """



@app.post("/add_wallpaper", dependencies=[Depends(authenticate)])
async def add_new_wallpaper(thumbnail_storage_file_name: str = Form(...), content_type: str = Form(...)):
    global img_names, img_emb
    data = {"success": False}

    if thumbnail_storage_file_name in [i[0] for i in img_names]:
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

    img_names.append((thumbnail_storage_file_name, content_type))
    img_emb_new = np.vstack((img_emb, batch_emb))

    with open(emb_path, "wb") as fOut:
        pickle.dump((img_names, img_emb_new), fOut)

    with open(emb_path, "rb") as fIn:
        img_names, img_emb = pickle.load(fIn)


    data["success"] = True
    data["message"] = "Wallpaper added successfully"

    return data

@app.get("/search", dependencies=[Depends(authenticate)])
def search_text(
    text: str,
    content_type: List[str] = Query(..., description="List of content types"),
    page: int = Query(1, description="Page number"),
    per_page: int = Query(10, description="Items per page")
):
    data = {"success": False}

    translated_query = translator.translate(text=text, dest='en').text

    inputs = tokenizer([translated_query], padding=True, return_tensors="pt")
    query_emb = model.get_text_features(**inputs)

    indexes = []

    for i, item in enumerate(img_names):
        if item[1] in content_type:
            indexes.append(i)

    result_img_emb = [img_emb[i] for i in indexes]
    result_img_names = [img_names[i] for i in indexes]

    corpus_embeddings = [torch.from_numpy(arr) for arr in result_img_emb]
    corpus_embeddings = torch.stack(corpus_embeddings)

    all_hits = util.semantic_search(query_emb, corpus_embeddings, top_k=per_page * page)[0]

    start_idx = (page - 1) * per_page
    end_idx = page * per_page
    hits = all_hits[start_idx:end_idx]

    images = [
        {
            "thumbnail_storage_file_name": result_img_names[hit["corpus_id"]][0],
            "content_type": result_img_names[hit["corpus_id"]][1],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{result_img_names[hit['corpus_id']][0]}",
            "score": f"{hit['score'] * 100:.2f}%",
        }
        for hit in hits if hit['score'] * 100 > 25.1
    ]
    data["success"] = True
    data["text_search"] = text
    data["data"] = images

    return data


@app.post("/search/image", dependencies=[Depends(authenticate)])
def search_image(content_type: List[str] = Query(..., description="List of content types"), page: int = 1, per_page: int = 10, image: UploadFile = File(...)):
    data = {"success": False}
    image = Image.open(image.file)

    images = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to("cpu")

    query_emb = model.get_image_features(images)

    indexes = []

    for i, item in enumerate(img_names):
        if item[1] in content_type:
            indexes.append(i)

    result_img_emb = [img_emb[i] for i in indexes]
    result_img_names = [img_names[i] for i in indexes]

    corpus_embeddings = [torch.from_numpy(arr) for arr in result_img_emb]
    corpus_embeddings = torch.stack(corpus_embeddings)

    all_hits = util.semantic_search(query_emb, corpus_embeddings, top_k=per_page * page)[0]

    start_idx = (page - 1) * per_page
    end_idx = page * per_page
    hits = all_hits[start_idx:end_idx]

    images = [
        {
            "thumbnail_storage_file_name": result_img_names[hit["corpus_id"]][0],
            "content_type": result_img_names[hit["corpus_id"]][1],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{result_img_names[hit['corpus_id']][0]}",
            "score": f"{hit['score'] * 100:.2f}%",
        }
        for hit in hits
    ]
    data["success"] = True
    data["data"] = images

    return data


@app.get("/recommend", dependencies=[Depends(authenticate)])
def recommend_images(thumbnail_storage_file_name: str, content_type: str = "single_image", page: int = 1, per_page: int = 10):
    global img_names, img_emb
    data = {"success": False}

    if thumbnail_storage_file_name in [i[0] for i in img_names]:
        idx = [i[0] for i in img_names].index(thumbnail_storage_file_name)
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
        img_names.append((thumbnail_storage_file_name, content_type))
        img_emb_new = np.vstack((img_emb, batch_emb))

        with open(emb_path, "wb") as fOut:
            pickle.dump((img_names, img_emb_new), fOut)

        with open(emb_path, "rb") as fIn:
            img_names, img_emb = pickle.load(fIn)

        idx = img_names.index(thumbnail_storage_file_name)
        query_emb = img_emb[idx]

    corpus_embeddings = [torch.from_numpy(arr) for arr in img_emb]
    corpus_embeddings = torch.stack(corpus_embeddings)

    all_hits = util.semantic_search(query_emb, corpus_embeddings, top_k=per_page * page)[0]

    start_idx = (page - 1) * per_page
    end_idx = page * per_page
    hits = all_hits[start_idx:end_idx]

    images = [
        {
            "image_name": img_names[hit["corpus_id"]][0],
            "content_type": img_names[hit["corpus_id"]][1],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{img_names[hit['corpus_id']][0]}",
            "score": f"{hit['score'] * 100:.2f}%"
        }
        for hit in hits if img_names[hit["corpus_id"]][0] != thumbnail_storage_file_name
    ]

    data["success"] = True
    data["data"] = images
    return data
