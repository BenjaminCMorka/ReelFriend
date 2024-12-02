import React, { useState } from 'react';
import { Card, CardContent } from './ui/card';
import Input from './ui/input';
import  Sidebar from './Sidebar';
import './styles/Browse.css';


import fallback from '../assets/movies-512.jpg';

const BrowsePage = () => {
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
      e.preventDefault(); 
      handleSearch(); 
    }
  };



  return (
    <div className="browse-page">
      <Sidebar />
      <div className="signin-container">
        <button className="sign-in-button">Sign In</button>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className=" mx-auto mt-16">


          <Card className="card">
            <CardContent className="card-content">
              <div className="search-container w-full">
                <Input 
                  type="text" 
                  placeholder="Search for Movies or Series..." 
                  className="input"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
              </div>
              
            </CardContent>
          </Card>

          {loading && <p>Loading...</p>}
          {error && <p style={{ color: 'red' }}>{error}</p>}
          {noResultsMessage && <p style={{ color: 'orange' }}>{noResultsMessage}</p>} {/* No results message */}

         
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

export default BrowsePage;
