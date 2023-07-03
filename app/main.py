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

app = FastAPI()
cache = TTLCache(maxsize=5000, ttl=600)
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



@app.get("/openapi.json")
async def get_open_api_endpoint():
    return get_openapi(title="API Document", version="1.0.0", routes=app.routes)


@app.get("/add_wallpaper")
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

@app.get("/search")
@cached(cache)
def search_text(text: str, num_results: int = 200):
    data = {"success": False}

    translated_query = translator.translate(text=text, dest='en').text

    inputs = tokenizer([translated_query], padding=True, return_tensors="pt")
    query_emb = model.get_text_features(**inputs)

    hits = util.semantic_search(query_emb, img_emb, top_k=num_results)[0]

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


@app.post("/search/image")
def search_image(num_results: int = 200, image: UploadFile = File(...)):
    data = {"success": False}
    image = Image.open(image.file)

    images = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to("cpu")

    query_emb = model.get_image_features(images)

    hits = util.semantic_search(query_emb, img_emb, top_k=num_results)[0]

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


@app.get("/recommend")
def recommend_images(thumbnail_storage_file_name: str, num_results: int = 200):
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

    hits = util.semantic_search(query_emb, img_emb, top_k=num_results)[0]

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
