
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import SignUpPage from './components/SignUp';
import SignInPage from './components/SignIn';
import TrendingPage from './components/TrendingPage';
import ProfilePage from './components/ProfilePage';
import Watchlist from './components/Watchlist';
import Browse from './components/Browse';
import Dashboard from './components/Dashboard';



function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/signup" element={<SignUpPage />} />
        <Route path="/signin" element={<SignInPage />} />
        <Route path="/trending" element={<TrendingPage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route path="/watchlist" element={<Watchlist/>} />
        <Route path="/browse" element={<Browse/>} />
        <Route path="/dashboard" element={<Dashboard/>} />
      </Routes>
    </Router>
  );
}

export default App;

