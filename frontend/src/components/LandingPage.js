import React from 'react';
import { useNavigate } from 'react-router-dom';
import './styles/LandingPage.css'; 





const LandingPage = () => {
  const navigate = useNavigate();

  
  const goToSignUp = () => {
    navigate('/signup'); 
  };

  const goToSignIn = () => {
    navigate('/signin'); 
  };
  return (
    <div className="landing-page">
      <div className="signin-container">
        <button className="sign-in-button" onClick={goToSignIn}>Sign In</button>
      </div>

      <div className="logo-container">
        <div className="logo">
        
          <h1 className="logo-title">flickFest</h1>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="max-w-4xl mx-auto mt-16">
          <h1 className="header">
            Your Gateway to Curated Viewing Experiences
          </h1>
          <p className="subheader">
          Start getting  tailored recommendations to build your perfect watchlist now!
          </p>


              <div className="sign-up-container">
                <button className="sign-up-button" onClick={goToSignUp}>Sign Up</button>
              </div>

        </div>
      </div>
    </div>
  );
};

export default LandingPage;
