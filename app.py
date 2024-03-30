from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import cv2
from keras.models  import load_model
import google.generativeai as genai
import re
import tensorflow as tf

GOOGLE_API_KEY = 'AIzaSyCB6FzLSYiuhOxJOxMC6C4UnB8DkwxwNFU'
genai.configure(api_key=GOOGLE_API_KEY)
Gmodel = genai.GenerativeModel('gemini-pro')
global clearLines
clearLines = []

label_map  = {
    '0': 'Healthy',
    '1': 'Powdery',
    '2': 'Rust',
}

label_map2 = {
    '0' :   "Apple Apple scab",
    '1' :	"Apple Black rot",
    '2' :	"Apple Cedar apple rust",
    '3' :	"Apple healthy",
    '4' :	"Bacterial leaf blight in rice leaf",
    '5' :	"Blight in corn Leaf",
    '6' :	"Blueberry healthy",
    '7' :	"Brown spot in rice leaf",
    '8' :	"Cercospora leaf spot",
    '9' :   "Cherry (including sour) Powdery mildew",
    '10':	"Cherry (including_sour) healthy",
    '11':	"Common Rust in corn Leaf",
    '12':	"Corn (maize) healthy",
    '13':	"Garlic",
    '14':	"Grape Black rot",
    '15':	"Grape Esca Black Measles",
    '16':	"Grape Leaf blight Isariopsis Leaf Spot",
    '17':	"Grape healthy",
    '18':	"Gray Leaf Spot in corn Leaf",
    '19':	"Leaf smut in rice leaf",
    '20':	"Orange Haunglongbing Citrus greening",
    '21':	"Peach healthy",
    '22':	"Pepper bell Bacterial spot",
    '23':	"Pepper bell healthy",
    '24':	"Potato Early blight",
    '25':	"Potato Late blight",
    '26':	"Potato healthy",
    '27':	"Raspberry healthy",
    '28':	"Sogatella rice",
    '29':	"Soybean healthy",
    '30':	"Strawberry Leaf scorch",
    '31':	"Strawberry healthy",
    '32':	"Tomato Bacterial spot",
    '33':	"Tomato Early blight",
    '34':	"Tomato Late blight",
    '35':	"Tomato Leaf Mold",
    '36':	"Tomato Septoria leaf spot",
    '37':	"Tomato Spider mites Two spotted spider mite",
    '38':	"Tomato Target Spot",
    '39':	"Tomato Tomato mosaic virus",
    '40':	"Tomato healthy",
    '41':	"algal leaf in tea",
    '42':	"anthracnose in tea",
    '43':	"bird eye spot in tea",
    '44':	"brown blight in tea",
    '45':	"cabbage looper",
    '46':	"corn crop",
    '47':	"ginger",
    '48':	"healthy tea leaf",
    '49':	"lemon canker",
    '50':	"onion",
    '51':	"potassium deficiency in plant",
    '52':	"potato crop",
    '53':	"potato hollow heart",
    '54':	"red leaf spot in tea",
    '55':	"tomato canker"
}

global model2
model2 = load_model("leaf_classifier.keras")

# json file
with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()
# loaded_mod = model_from_json(loaded_model_json)
loaded_mod = tf.keras.models.model_from_json(loaded_model_json)
loaded_mod.load_weights("model_weights.h5")
loaded_mod.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])


# global model3
# model3 = load_model("leaf_classifier2 (6).keras")

app = Flask(__name__)

with open("rf_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)
    
    
@app.route('/')
def main_p():
    # Render your main page template here
    return render_template("/signup.html")

@app.route("/signin") #here
def signin():

     if request.method == "POST":
         email = request.form.get("email")
         password = request.form.get("password")



     # Redirect to sign-in page if sign-in fails
     return render_template("/signin.html")

@app.route("/main")
def main_page():
    # Render your main page template here
    return render_template("/home.html")


    
@app.route('/disease_detection')
def detection():
    return render_template('diseaseDetect.html')

@app.route("/diseasedetection", methods=["POST"])
def predict_disease():
    if request.method == "POST":
        if request.files:
            symptoms = request.form.get("symptoms")
            crop = request.form.get("crop")

            image = request.files["image"].read()
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_image = img.astype("float32") / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)

            pred = model2.predict(processed_image)[0].argmax()

            pred2 = loaded_mod.predict(processed_image)[0].argmax()
            label = label_map.get(str(pred))

            if label == "Healthy":
                label2 = "Healthy"
                clearLines = ["The plant is healthy. No disease detected."]
            else:
                label2 = label_map2.get(str(pred2))
                prompt = f"Based on the symptoms {symptoms} and {label} and crop {crop} predict the disease and suggest the cure."
                response = Gmodel.generate_content(prompt)
                # Split the response text into a list of lines
                lines = response.text.splitlines()

                # Clean each line using regular expression
                clearLines = [re.sub(r"[^\w\s\!\?\.\,]", "", line) for line in lines]

    return render_template('diseaseDetect.html', prediction1=label, diseaseprediction=clearLines)

@app.route('/recommendation')
def recomendation():
    return render_template('plantRec.html')

@app.route('/croprecommendation', methods=['POST'])
def predict_crops():
  if request.method == 'POST':
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    user_text = request.form['doubt']

    prompt = f"Based on given soil nutrients and environmental conditions, suggest suitable crops:\n\n Soil Nutrients: N={N}, P={P}, K={K}\n Environmental Conditions: Temperature={temperature}, Humidity={humidity}, pH={ph}, Rainfall={rainfall} and also have doubt {user_text}"
    response = Gmodel.generate_content(prompt)

    # Split the response text into a list of lines
    lines = response.text.splitlines()

    # Clean each line using regular expression
    clean_lines = [re.sub(r"[^\w\s\!\?\.\,]", "", line) for line in lines]
    print(clean_lines)

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_data_scaled = loaded_scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    # Extract the predicted crop from the list (assuming it has only one element)
    predicted_crop = prediction[0]

    return render_template('plantRec.html', prediction=clean_lines, pre=predicted_crop)


if __name__ == "__main__":
    app.run(debug=False)