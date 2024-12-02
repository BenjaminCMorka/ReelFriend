import React, { useEffect, useState } from 'react';
import  Sidebar from './Sidebar';
import './styles/LandingPage.css';
import './styles/MovieSearch.css';
import fallback from '../assets/movies-512.jpg';

const TrendingPage = () => {
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const apiKey = '7a0553e66258137e7f70085c7dde6cbc'; 

  useEffect(() => {
    const fetchTrendingMovies = async () => {
      try {
        const response = await fetch(`https://api.themoviedb.org/3/trending/movie/week?api_key=${apiKey}`);
        if (!response.ok) throw new Error('Failed to fetch trending movies');
        
        const data = await response.json();
        if (data.results) {
          // Shuffle and select 10 random movies
          const shuffledMovies = data.results.sort(() => 0.5 - Math.random());
          setResults(shuffledMovies.slice(0, 10)); 
        }
      } catch (err) {
        setError('Could not load trending movies. Please try again later.');
      }
    };

    fetchTrendingMovies();
  }, []);

  return (
    <div className="trending-page">
      <Sidebar />
      <div className="signin-container">
        <button className="sign-in-button">Sign In</button>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="max-w-4xl mx-auto mt-16">
          <h1 className="header">
            Trending Flicks
          </h1>
         
          {error && <p style={{ color: 'red' }}>{error}</p>}

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

export default TrendingPage;
