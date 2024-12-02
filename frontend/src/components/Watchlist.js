import React from 'react';

import  Sidebar from './Sidebar';
import './styles/LandingPage.css';


const RoomsLobby = () => {

  return (
    <div className="watchlist">
      <Sidebar />
      <div className="signin-container">
        <button className="sign-in-button">Sign In</button>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="max-w-4xl mx-auto mt-16">
          <h1 className="header">
            Welcome to your Watchlist
          </h1>

          
        </div>
      </div>
    </div>
  );
};

export default RoomsLobby;
