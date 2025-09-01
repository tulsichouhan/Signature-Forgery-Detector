from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For sessions

# Custom function for Siamese model
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

tf.keras.utils.get_custom_objects()["euclidean_distance"] = euclidean_distance

# Load model
MODEL_PATH = "model/siamese_model.h5"
siamese_model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"euclidean_distance": euclidean_distance}, compile=False)

IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE) / 255.0
    return img.reshape(1, 128, 128, 1)

def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/", methods=["GET", "POST"])
def index():
    if "user_id" not in session:
        return redirect("/login")
    if request.method == "POST":
        if "image1" not in request.files or "image2" not in request.files:
            return jsonify({"error": "Please upload both images!"})

        image1 = request.files["image1"]
        image2 = request.files["image2"]

        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        image1_path = os.path.join(upload_folder, image1.filename)
        image2_path = os.path.join(upload_folder, image2.filename)

        image1.save(image1_path)
        image2.save(image2_path)

        img1 = preprocess_image(image1_path)
        img2 = preprocess_image(image2_path)

        if img1 is None or img2 is None:
            return jsonify({"error": "Invalid image format!"})

        similarity_score = siamese_model.predict([img1, img2])[0][0]
        threshold = 0.4
        result = "Match (Same Person)" if similarity_score < threshold else "Forgery (Different Person)"

        return jsonify({"similarity_score": float(similarity_score), "result": result})

    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password)).fetchone()
        conn.close()

        if user:
            session["user_id"] = user["id"]
            return redirect("/")
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
            conn.commit()
            conn.close()
            return redirect("/login")
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already exists.")

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)
