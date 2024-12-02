import React, { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import './styles/Dashboard.css';
import { Card, CardContent } from './ui/card';
import axios from 'axios'

import Input from './ui/input';
import fallback from '../assets/movies-512.jpg';

const Dashboard = () => {
  const [showOnboarding, setShowOnboarding] = useState(true); 
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedGenres, setSelectedGenres] = useState([]); 
  const [favMovies, setFavMovies] = useState([]);

  const [startYear, setStartYear] = useState('');
  const [endYear, setEndYear] = useState('2023');
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [noResultsMessage, setNoResultsMessage] = useState('');
  const [username, setUsername] = useState('');
  


  const questions = [
    'What Genres Do You Enjoy The Most?',
    'Preferred Release Period for Movies?',
    'What Are Your Favorite Movies Currently?)',
  ];

  const genres = [
    'Action', 'Comedy', 'Drama', 'Horror', 'Romance', 
    'Sci-Fi', 'Thriller', 'Adventure', 'Fantasy', 'Documentary', 'Mystery', 'Animation'
  ];

  useEffect(() => {
    // Fetch user data when component mounts
    axios.get('/api/auth')
      .then(response => {
        setUsername(response.data.username);
      })
      .catch(error => console.error('Error fetching user data:', error));
  }, []);



  useEffect(() => {
    const fetchUserStatus = async () => {
      try {
        const response = await axios.get('/api/auth/status'); 
        if (response.data.hasOnboarded) {
          setShowOnboarding(false);
        } else {
          setShowOnboarding(true);
        }
      } catch (err) {
        console.error("Error fetching user onboarding status", err);
        setShowOnboarding(true); 
      }
    };

    fetchUserStatus();
  }, []);



  const toggleGenreSelection = (genre) => {
    setSelectedGenres(prevSelectedGenres =>
      prevSelectedGenres.includes(genre)
        ? prevSelectedGenres.filter(g => g !== genre) 
        : [...prevSelectedGenres, genre] 
    );
  };

  const handleNext = () => {
    if (currentStep < questions.length - 1) {
      setCurrentStep(prevStep => prevStep + 1);
    } else {
      setShowOnboarding(false);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(prevStep => prevStep - 1);
    }
  };

  const handleYearChange = (e) => {
    const { name, value } = e.target;
    if (name === 'startYear') {
      setStartYear(value);
    } else if (name === 'endYear') {
      setEndYear(value);
    }
  };

  const handleSearch = async () => {
    setLoading(true);
    setError('');
    setNoResultsMessage('');
    setResults([]); 


   
    if (!searchTerm.trim()) {
        setNoResultsMessage("Please Enter Movie Title."); 
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
    <div className="dashboard">
      <Sidebar />

      <div className="main-content max-w-4xl mx-auto mt-16">


        {showOnboarding && (
          <div className="onboarding-modal">
            <div className="modal-content">
              <h1>Let's Get To Know You Better...</h1>
              <p className="question-prompt">{questions[currentStep]}</p>

              {currentStep === 0 && (
                <div className="genre-selection">
                  {genres.map(genre => (
                    <div
                      key={genre}
                      className={`genre-box ${selectedGenres.includes(genre) ? 'selected' : ''}`}
                      onClick={() => toggleGenreSelection(genre)}
                    >
                      {genre}
                    </div>
                  ))}
                </div>
              )}

                {currentStep === 1 && (
                <div className="year-input">
                  <input
                    type="number"
                    name="startYear"
                    value={startYear}
                    onChange={handleYearChange}
                    placeholder="Start Year"
                    className="year-input-box"
                  />
                  <span className="to-text">To</span>
                  <input
                    type="number"
                    name="endYear"
                    value={endYear}
                    onChange={handleYearChange}
                    placeholder="End Year"
                    className="year-input-box"
                  />
                </div>
              )}

              {currentStep === 2 && (
               <div className="favorite-search">
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
               {noResultsMessage && <p style={{ color: 'orange' }}>{noResultsMessage}</p>}
             
               {/* Render selected movies */}
               <div className="selected-movies">
                 {favMovies.map((movie, index) => (
                   <div key={index} className="selected-movie-box">
                     <span>{movie.title || movie.name}</span>
                     <button
                       className="remove-movie-btn"
                       onClick={() => setFavMovies((prev) => prev.filter((_, i) => i !== index))}
                     >
                       ✖
                     </button>
                   </div>
                 ))}
               </div>
             
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
                         href="#"
                         onClick={(e) => {
                           e.preventDefault();
                           setFavMovies((prev) => [...prev, item]);
                           setSearchTerm('');
                           setResults([]);
                         }}
                         className="watch-now"
                       >
                         Select
                       </a>
                     </div>
                   </div>
                 ))}
               </div>
             </div>
                           )}

              

<div className="progress-indicator">
                <p>{currentStep + 1} / {questions.length}</p>
            </div>

              <div className="buttons-container">
                <button className="prev-button" onClick={handlePrev}>
                  &#8592; {/* Left arrow character */}
                </button>
                <button className="next-button" onClick={handleNext}>{currentStep !== questions.length - 1 ? 'Next' : 'Finish'}</button>
              </div>
            </div>

          </div>
        )}
        
        {!showOnboarding && (
          <div className="content">
            <p>Welcome to your dashboard! Explore your recommendations and watchlist here.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;