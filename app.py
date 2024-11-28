from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
from keras.models import load_model
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
app.secret_key = "AzxSAzXsAaZxS"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load your saved model and tokenizer
model_name_or_path = "models/chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# Load your models
sugarcane_model = load_model('models/sugarcane.keras')
grapes_model = load_model('models/grapes_model.keras')

def model_prediction(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((244, 244))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Debug session contents
        app.logger.info(f"Session contents during prediction: {session}")

        # Ensure a crop is selected
        selected_crop = session.get('selected_crop')
        if not selected_crop:
            return "Error", "No crop selected", 0

        # Predict with the corresponding model
        if selected_crop == 'sugarcane':
            predictions = sugarcane_model.predict(image)
            disease_labels = {
                0: "Banded Chlorosis", 1: "Brown Rust", 2: "Brown Spot", 
                3: "Dried Leaves", 4: "Grassy Shoot", 5: "Healthy", 
                6: "Pokkah Boeng", 7: "Sett Rot", 8: "Smut", 
                9: "Viral Disease", 10: "Yellow Leaf"
            }
        elif selected_crop == 'grapes':
            predictions = grapes_model.predict(image)
            disease_labels = {
                0: "Black Rot", 1: "ESCA", 2: "Healthy", 
                3: "Leaf Blight"
            }
        else:
            return "Error", "Invalid crop selection", 0

        # Determine disease and confidence
        disease_idx = np.argmax(predictions)
        confidence = np.max(predictions)

        # Add confidence threshold
        if confidence < 0.5:
            return selected_crop.capitalize(), "Uncertain prediction", confidence

        disease = disease_labels.get(disease_idx, "Unknown Disease")
        return selected_crop.capitalize(), disease, confidence
    except Exception as e:
        return f"Error {e}", f"Error {e}", 0


# @app.route('/set_crop', methods=['POST'])
# def set_crop():
#     selected_crop = request.form.get('crop')  # Get the selected crop from the form
#     app.logger.info(f"Received crop selection: {selected_crop}")

#     if not selected_crop:
#         if session.get('role') == 'admin': # Assuming session tracks user role
#             flash("Please select a crop before proceeding.")
#             return redirect(url_for('admin_dashboard'))
#         else:
#             flash("Please select a crop before proceeding.")
#             return redirect(url_for('user_dashboard'))

#     # Save crop to session
#     session['selected_crop'] = selected_crop
#     app.logger.info(f"Crop saved in session: {session['selected_crop']}")

#     flash(f"You selected {selected_crop.capitalize()}. You can now upload an image.")
#     return redirect(url_for('upload_file'))

@app.route('/set_crop', methods=['POST'])
def set_crop():
    data = request.get_json()
    selected_crop = data.get('crop')
    if not selected_crop:
        return jsonify({"success": False, "message": "No crop selected."}), 400
    
    session['selected_crop'] = selected_crop
    return jsonify({"success": True, "message": f"Crop '{selected_crop}' selected."})


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "user" and password == "user":
            session["role"] = "user"
            return redirect(url_for("user_dashboard"))
        elif username == "admin" and password == "admin":
            session["role"] = "admin"
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid username or password. Please try again.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")



@app.route("/user_dashboard", methods=["GET", "POST"])
def user_dashboard():
    app.logger.info(f"Session contents at dashboard: {session}")
    if session.get("role") != "user":
        return redirect(url_for("login"))

    if request.method == "POST":
        return redirect(url_for("upload_file"))

    return render_template("user_dashboard.html")


@app.route("/admin_dashboard", methods=["GET", "POST"])
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login"))

    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload":
            return redirect(url_for("upload_image"))
        elif action == "train":
            return redirect(url_for("train_model"))

    return render_template("admin_dashboard.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.info(f"Crop selected for upload: {session.get('selected_crop')}")
    if 'file' not in request.files:
        return render_template('upload.html', image_path=None, class_name=None, disease=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', image_path=None, class_name=None, disease=None)

    try:
        # Save the captured photo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        plant, disease, confidence = model_prediction(file_path)

        if session.get('role') == 'admin':  # Assuming session tracks user role
            if 'selected_crop' not in session:
                flash("Please select a crop before uploading an image.")
                app.logger.info("No crop selected in session; redirecting to dashboard.")
            redirect_url = url_for('admin_dashboard')
        else:
            if 'selected_crop' not in session:
                flash("Please select a crop before uploading an image.")
                app.logger.info("No crop selected in session; redirecting to dashboard.")
            redirect_url = url_for('user_dashboard')

        return render_template('upload.html', image_path=file.filename, class_name=plant, disease=f"{disease} ({confidence*100:.2f}%)", redirect_url=redirect_url)
    except Exception as e:
        return render_template('upload.html', image_path=None, class_name=None, disease=None, error=str(e))


# @app.route("/train_model", methods=["GET", "POST"])
# def train_model():
#     if session.get("role") != "admin":
#         return redirect(url_for("login"))

#     if request.method == "POST":
#         crop_name = request.form["crop_name"]
#         num_classes = int(request.form["num_classes"])
#         # Add code to handle image uploads and training
#         flash("Training started. This will take some time.")
#         return redirect(url_for("admin_dashboard"))

#     return render_template("train_model.html")

@app.route('/chatbot')
def chatbot_page():
    if session.get('role') == 'admin':  # Assuming session tracks user role
        redirect_url = url_for('admin_dashboard')
    else:
        redirect_url = url_for('user_dashboard')
    return render_template('chatbot.html', redirect_url=redirect_url)  # Renders chatbot.html from the templates folder

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return {"error": "No message provided"}, 400

        # Tokenize the user input
        inputs = tokenizer(user_input, return_tensors="pt")

        # Generate a response
        # outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            max_length=100,  
            num_beams=5,     
            temperature=0.3, 
            no_repeat_ngram_size=2, 
            early_stopping=True, 
        )
        # Decode the generated text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Ensure only the response is returned
        return {"response": response.split(":")[-1].strip()}, 200

    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'GET':
        return render_template('train_model.html')

    if request.method == 'POST':
        crop_name = request.form.get('crop_name')
        num_classes = int(request.form.get('num_classes'))
        class_names = []

        # Validate and save files
        crop_folder = os.path.join(UPLOAD_FOLDER, crop_name)
        os.makedirs(crop_folder, exist_ok=True)

        for i in range(1, num_classes + 1):
            class_name = request.form.get(f'class_name_{i}')
            class_folder = os.path.join(crop_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)

            files = request.files.getlist(f'class_images_{i}')
            if len(files) < 5:
                flash(f"Please upload at least 5 images for class {class_name}.")
                return redirect(url_for('train_model'))

            for file in files:
                if file:
                    file.save(os.path.join(class_folder, file.filename))

            class_names.append(class_name)

        flash(f"Model data for {crop_name} successfully uploaded and organized!")
        return redirect(url_for('train_model'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error_message="Page not found."), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
