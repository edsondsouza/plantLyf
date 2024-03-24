from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import cv2
from keras.models import load_model

label_map = {
    '0': 'Healthy',
    '1': 'Powdery',
    '2': 'Rust',
    # Add more entries as needed
}

global model2
model2 = load_model("leaf_classifier.keras")

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
    return render_template("/main.html")


    
@app.route('/disease_detection')
def detection():
    return render_template('diseaseDetect.html')


@app.route("/diseasedetection", methods=["POST"])
def predict_disease():
    if request.method == "POST":
        if request.files:
            image = request.files["image"].read()
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_image = img.astype("float32") / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)

            pred = model2.predict(processed_image)[0].argmax()

            label = label_map.get(str(pred))

    return str(label)



@app.route('/recommendation')
def recomendation():
    return render_template('plantRec.html')

@app.route('/croprecommendation', methods=['POST'])
def predict_crops():
    if request.method == 'POST':
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_data_scaled = loaded_scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)

    return render_template('plantRec.html', prediction=prediction[0])



if __name__ == "__main__":
    app.run(debug=True)