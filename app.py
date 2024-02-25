from flask import Flask, render_template, request, redirect, url_for, session
import firebase_admin
from firebase_admin import auth 
app = Flask(__name__)
@app.route('/')
def main_p():
    # Render your main page template here
    return render_template("/signup.html")

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

@app.route("/main")
def main_page():
    # Render your main page template here
    return render_template("/main.html")

@app.route('/croprecommendation')
def recommendation():
    return render_template('plantRec.html')
@app.route('/disease_detection')
def detection():
    return render_template('diseaseDetect.html')

if __name__ == "__main__":
    app.run(debug=True)