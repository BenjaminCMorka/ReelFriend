import React, { useState } from 'react';
import './styles/MovieSearch.css'; 

const MovieSearch = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [movies, setMovies] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [noResultsMessage, setNoResultsMessage] = useState(''); // Define noResultsMessage state

    const handleSearch = async () => {
        // Clear previous results and messages
        setMovies([]);
        setNoResultsMessage('');
        
        if (!searchTerm.trim()) {
            setNoResultsMessage('Please enter a search term.'); 
            return;
        }

        setLoading(true);
        setError('');
        try {
            const response = await fetch(`https://api.themoviedb.org/3/search/movie?api_key=7a0553e66258137e7f70085c7dde6cbc&query=${searchTerm}`);
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();

            // Check if there are any results
            if (data.results && data.results.length > 0) {
                setMovies(data.results);
            } else {
                setNoResultsMessage(`No results found for "${searchTerm}".`); // Set message for no results
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
        <div>
            <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={handleKeyDown} // Use onKeyDown instead of onKeyPress
                placeholder="Search for a movie..."
            />
            <button onClick={handleSearch}>Search</button>

            {loading && <p>Loading...</p>}
            {error && <p style={{ color: 'red' }}>{error}</p>}
            {noResultsMessage && <p style={{ color: 'orange' }}>{noResultsMessage}</p>} {/* No results message */}

            <div className="movie-cards">
                {movies.map((movie) => (
                    <div key={movie.id} className="movie-card">
                        <img src={`https://image.tmdb.org/t/p/w200${movie.poster_path}`} alt={movie.title} />
                        <h3>{movie.title}</h3>
                        <p>{movie.overview}</p>
                        <a href={`https://www.themoviedb.org/movie/${movie.id}`} target="_blank" rel="noopener noreferrer">Watch Now</a>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MovieSearch;
