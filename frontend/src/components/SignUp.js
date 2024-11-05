import React from 'react';
import './styles/SignUp.css'; // Import your CSS file
import logo from '../assets/flickfest_logo.png'; // Adjust the path as needed

const SignUp = () => {
  return (
    <div className="signup-page">
      <div className="signup-container">
        <div className="logo-container">
          <img src={logo} alt="FlickFest Logo" className="logo-icon" />
          <h1 className="logo-title">flickFest</h1>
        </div>
        <h2>Sign Up</h2>
        <button className="google-signup-button">Sign up with Google</button>
        <form className="signup-form">
          <input type="text" placeholder="Email" required />
          <input type="password" placeholder="Password" required />
          <button type="submit" className="signup-submit-button">Create Account</button>
        </form>
      </div>
    </div>
  );
};

export default SignUp;
