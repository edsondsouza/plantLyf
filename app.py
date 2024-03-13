from flask import Flask, render_template, request, redirect, url_for, session
import firebase_admin
import numpy as np
import pickle
from firebase_admin import auth 
app = Flask(__name__)
with open("rf_model.pkl", "rb") as file:
    model = pickle.load(file)
    

# signup page
@app.route('/')
def main_p():
    # Render your main page template here
    return render_template("/signup.html")

# signin page
@app.route("/signin") #here
def signin():
     if request.method == "POST":
         email = request.form.get("email")
         password = request.form.get("password")
         try:
             user = firebase_admin.auth.sign_in_with_email_and_password(email, password)
             # Handle sign-in success, update user data in database, redirect, etc.
             return render_template("/main.html")
         except firebase_admin.auth.InvalidEmailError:
             # Handle invalid email error
             pass
         except firebase_admin.auth.InvalidPasswordError:
             # Handle invalid password error
             pass
         except firebase_admin.auth.UserNotFoundError:
             # Handle user not found error
             pass
         except firebase_admin.auth.AuthError as e:
             # Handle other authentication errors
             pass
     # Redirect to sign-in page if sign-in fails
     return render_template("/signin.html")

# main page
@app.route("/main")
def main_page():
    # Render your main page template here
    return render_template("/main.html")

# disease detection page
@app.route('/diseasedetection')
def detection():
    return render_template('diseaseDetect.html')

# crop recommendation page
@app.route('/recommendation')
def recommendation():
    return render_template('plantRec.html')

@app.route('/croprecommendation', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)

        return render_template('plantRec.html', prediction=prediction[0])


# about page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)