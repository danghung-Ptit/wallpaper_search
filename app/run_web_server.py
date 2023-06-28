from flask import Flask, request, jsonify
from PIL import Image
from sentence_transformers import util
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from googletrans import Translator
import requests
import pickle
import numpy as np

app = Flask(__name__)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

emb_path = "app/wallpaper_indexing/wallpaper-embeddings.pkl"

with open(emb_path, "rb") as fIn:
    img_names, img_emb = pickle.load(fIn)


@app.route("/")
def homepage():
	"""
	Endpoint for homepage
	"""
	return "Welcome to the Recommed Wallpaper V2 REST API!"

@app.route("/search", methods=["GET"])
def search_text():
    data = {"success": False}
    option = "search"
    query = request.args.get("text")
    translator = Translator()
    input = translator.translate(text=query, dest='en')
    input = input.text


    inputs = tokenizer([input], padding=True, return_tensors="pt")
    query_emb = model.get_text_features(**inputs)

    hits = util.semantic_search(query_emb, img_emb, top_k=50)[0]

    images = [
        {
            "thumbnail_storage_file_name": img_names[hit["corpus_id"]],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{img_names[hit['corpus_id']]}",
            "score": f"{hit['score']*100:.2f}%",
        }
        for hit in hits
    ]
    data["success"] = True
    data["text_search"] = query
    data["data"] = images

    return jsonify(data)


@app.route("/search/image", methods=["POST"])
def search_image():
    image_file = request.files["image"]
    image = Image.open(image_file)

    images = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to("cpu")

    query_emb = model.get_image_features(images)

    hits = util.semantic_search(query_emb, img_emb, top_k=50)[0]

    images = [
        {
            "image_name": img_names[hit["corpus_id"]],
            "score": f"{hit['score']*100:.2f}%",
        }
        for hit in hits
    ]

    return jsonify(images)


@app.route("/recommend", methods=["GET"])
def recommend_images():
    global img_names, img_emb
    data = {"success": False}
    option = "recommend"
    image_path = request.args.get("image_path")
    if image_path in img_names:
        idx = img_names.index(image_path)
        query_emb = img_emb[idx]
    else:
        data["message"] = "Image not learn!"
        return jsonify(data)

    #     try:
    #         image = Image.open(
    #             requests.get(f"https://wallpapernew.net/api/static/images/storage/{image_path}",
    #                          stream=True).raw).convert('RGB')
    #     except Exception as e:
    #         print(f"Error loading image: {str(e)}")
    #         return jsonify(data)
    #
    #     images = processor(
    #         text=None,
    #         images=image,
    #         return_tensors='pt'
    #     )['pixel_values'].to("cpu")
    #
    #     query_emb = model.get_image_features(images)
    #     batch_emb = query_emb.squeeze(0).cpu().detach().numpy()
    #
    #     img_names.append(image_path)
    #     img_emb_new = np.vstack((batch_emb, img_emb))
    #
    #     with open(emb_path, "wb") as fOut:
    #         pickle.dump((img_names, img_emb_new), fOut)
    #
    #     with open(emb_path, "rb") as fIn:
    #         img_names, img_emb = pickle.load(fIn)
    #
    # idx = img_names.index(image_path)
    # query_emb = img_emb[idx]

    hits = util.semantic_search(query_emb, img_emb, top_k=50)[0]

    images = [
        {
            "image_name": img_names[hit["corpus_id"]],
            "paths": f"https://wallpapernew.net/api/static/images/storage/{img_names[hit['corpus_id']]}",
            "score": f"{hit['score']*100:.2f}%",
            "id": hit["corpus_id"]
        }
        for hit in hits
    ]

    data["success"] = True
    data["data"] = images

    return jsonify(data)


if __name__ == "__main__":
    app.run()
