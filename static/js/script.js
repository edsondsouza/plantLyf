// import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.2/firebase-app.js";
//     import {
//       getDatabase,
//       update,
//       ref,
//     } from "https://www.gstatic.com/firebasejs/10.7.2/firebase-database.js";
//     import {
//       getAuth,
//       signInWithEmailAndPassword,
//     } from "https://www.gstatic.com/firebasejs/10.7.2/firebase-auth.js";

//     // Retrieve the Firebase configuration from localStorage
//     const firebaseConfig = JSON.parse(localStorage.getItem("firebaseConfig"));

//     // Initialize Firebase
//     const app = initializeApp(firebaseConfig);
//     const database = getDatabase(app);
//     const auth = getAuth();

//     const signin = document.querySelector("#signin");

//     signin.addEventListener("click", (e) => {
//       const email = document.querySelector('#email').value;
//       const password = document.querySelector('#password').value;

//       e.preventDefault();
//       // console.log('sign in clicked')
//       signInWithEmailAndPassword(auth, email, password)
//         .then((userCredential) => {
//           // Signed in
//           const user = userCredential.user;

//           const dt = new Date();
//           update(ref(database, "users/" + user.uid), {
//             last_login: dt
//             // never save password to the db
//           });
//           console.log('user logged in!')
//           window.location.href = '../pages/main.html'
//         })
//         .catch((error) => {
//           const errorCode = error.code;
//           const errorMessage = error.message;
//         });
//     });