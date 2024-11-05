import React, { useState } from 'react';
import { User, Home, TrendingUp, Film } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import Button from './ui/button';
import Input from './ui/input';
import './styles/LandingPage.css'; 
import './styles/MovieSearch.css'; 
import logo from '../assets/flickfest_logo.png';
import fallback from '../assets/movies-512.jpg';

const LandingPage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [noResultsMessage, setNoResultsMessage] = useState('');

  const handleSearch = async () => {
    setLoading(true);
    setError('');
    setNoResultsMessage('');
    setResults([]); 

   
    if (!searchTerm.trim()) {
        setNoResultsMessage("Please enter a search term."); 
        setLoading(false); 
        return; 
    }

    const apiKey = '7a0553e66258137e7f70085c7dde6cbc'; 
    const url = `https://api.themoviedb.org/3/search/multi?api_key=${apiKey}&query=${searchTerm}`;

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Network response was not ok');
        
        const data = await response.json();

        // Check if there are results from the API
        if (data.results && data.results.length > 0) {
            setResults(data.results);
        } else {
            setNoResultsMessage(`No results found for "${searchTerm}".`); // Set message for no results found
        }
    } catch (err) {
        setError('Failed to fetch movies. Please try again later.');
    } finally {
        setLoading(false);
    }
};

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault(); // Prevent form submission if inside a form
      handleSearch(); // Trigger search on Enter key press
    }
  };

  return (
    <div className="landing-page">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="logo-container">
          <div className="logo">
            <img src={logo} alt="FlickFest Logo" className="logo-icon" />
            <h1 className="logo-title">flickFest</h1>
          </div>
        </div>
        
        <nav className="nav">
          <Button className="nav-button">
            <Home className="icon" />
            Home
          </Button>
          <Button className="nav-button">
            <TrendingUp className="icon" />
            Trending 
          </Button>
          <Button className="nav-button">
            <Film className="icon" />
            For You
          </Button>
          <Button className="nav-button">
            <User className="icon" />
            My Profile
          </Button>
        </nav>
      </div>

      <div className="signin-container">
        <button className="sign-in-button">Sign In</button>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="max-w-4xl mx-auto mt-16">
          <h1 className="header">
            Your Gateway to Curated Viewing Experiences
          </h1>
          <p className="subheader">
            Discover a handpicked selection of movies and shows tailored to your preferences. 
            Connect with friends to share your viewing experiences and recommendations!
          </p>

          <Card className="card">
            <CardContent className="card-content">
              <div className="search-container">
                <Input 
                  type="text" 
                  placeholder="Search for Movies or Series..." 
                  className="input"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
              </div>
              <div className="sign-up-container">
                <button className="sign-up-button">Sign Up</button>
              </div>
            </CardContent>
          </Card>

          {loading && <p>Loading...</p>}
          {error && <p style={{ color: 'red' }}>{error}</p>}
          {noResultsMessage && <p style={{ color: 'orange' }}>{noResultsMessage}</p>} {/* No results message */}

          {/* Display Search Results */}
          <div className="movie-cards">
            {results.map((item) => (
              <div key={item.id} className="movie-card">
                <img 
                  src={item.poster_path 
                    ? `https://image.tmdb.org/t/p/w200${item.poster_path}` 
                    : fallback} 
                  alt={item.title || item.name} 
                  className="movie-image" 
                />
                <div className="movie-details">
                  <h3>{item.title || item.name}</h3>
                  <p>{item.overview}</p>
                  <a 
                    href={`https://www.themoviedb.org/movie/${item.id}`} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="watch-now"
                  >
                    Watch Now
                  </a>
                </div>
              </div>
            ))}
          </div>

        </div>
      </div>
    </div>
  );
};

export default LandingPage;
