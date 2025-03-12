import { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import axios from "axios";
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { toast } from 'react-hot-toast';

// Genre mapping from SearchResultsPage


const FindSimilarPage = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [movieDetails, setMovieDetails] = useState([]);
  const [matchedTitle, setMatchedTitle] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [trailers, setTrailers] = useState({});
  const [showDescription, setShowDescription] = useState({});
  const { user, isAuthenticated, addToWatchlist } = useAuthStore();
  const navigate = useNavigate();

  // Function to fetch movie details for each TMDB ID
  const fetchMovieDetails = async (movieIds) => {
    const apiKey = '7a0553e66258137e7f70085c7dde6cbc';
    const movieDetailsArray = [];
    
    for (const id of movieIds) {
      try {
        const response = await fetch(
          `https://api.themoviedb.org/3/movie/${id}?api_key=${apiKey}`
        );
        if (response.ok) {
          const data = await response.json();
          if (data) {
            movieDetailsArray.push(data);
            fetchTrailer(id);
          }
        }
      } catch (err) {
        console.error(`Error fetching details for movie ${id}:`, err);
      }
    }
    
    return movieDetailsArray;
  };

  // Function to fetch trailers (copied from SearchResultsPage)
  const fetchTrailer = async (movieId) => {
    const apiKey = '7a0553e66258137e7f70085c7dde6cbc';
    const url = `https://api.themoviedb.org/3/movie/${movieId}/videos?api_key=${apiKey}`;

    try {
      const response = await fetch(url);
      const data = await response.json();
      if (data.results && data.results.length > 0) {
        const trailer = data.results.find(video => video.type === 'Trailer');
        if (trailer) {
          setTrailers(prevState => ({
            ...prevState,
            [movieId]: `https://www.youtube.com/watch?v=${trailer.key}`,
          }));
        }
      }
    } catch (err) {
      console.error('Error fetching trailer:', err);
    }
  };

  // Fetch movie details whenever recommendations change
  useEffect(() => {
    if (recommendations.length > 0) {
      const getMovieDetails = async () => {
        const details = await fetchMovieDetails(recommendations);
        setMovieDetails(details);
      };
      getMovieDetails();
    } else {
      setMovieDetails([]);
    }
  }, [recommendations]);

  const handleSearch = async (e) => {
    e.preventDefault();
  
    if (!searchTerm.trim()) return;
  
    setIsLoading(true);
    setError("");
    setMatchedTitle(""); // Clear previous closest match
    setMovieDetails([]);
  
    try {
      const response = await axios.post("http://localhost:5001/api/recommender", {
        movieTitle: searchTerm,
        count: 10
      });
      
      if (response.data.error) {
        setError(response.data.error);
        setRecommendations([]);
        setMatchedTitle("");
      } else {
        setRecommendations(response.data.recommendations || []);
        setMatchedTitle(response.data.title_matched || "");
      }
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      setError(err.response?.data?.error || "Failed to get recommendations");
      setRecommendations([]);
      setMatchedTitle("");
    } finally {
      setIsLoading(false);
    }
  };
  

  // Toggle description function (copied from SearchResultsPage)
  const toggleDescription = (movieId) => {
    setShowDescription(prevState => ({
      ...prevState,
      [movieId]: !prevState[movieId],
    }));
  };

  // Add to watchlist function (copied from SearchResultsPage)
  const handleAddToWatchlist = async (movie) => {
    if (!isAuthenticated) {
      toast.error('Please login to add movies to your watchlist');
      navigate('/login');
      return;
    }

    if (user?.watchlist?.includes(movie.id.toString())) {
      toast.error(`"${movie.title}" is already in your watchlist!`);
      return;
    }

    try {
      const result = await addToWatchlist(
        movie.id.toString(), 
        movie.title,
        movie.poster_path
      );
      
      if (result.success === false && result.message === "Movie already in watchlist") {
        toast.error(`"${movie.title}" is already in your watchlist!`);
      } else {
        toast.success(`Added "${movie.title}" to your watchlist!`);
      }
    } catch (err) {
      console.error('Error adding to watchlist:', err);
      toast.error('Failed to add to watchlist. Please try again.');
    }
  };

  // Helper function to check if a movie is in watchlist
  const isInWatchlist = (movieId) => {
    return user?.watchlist?.includes(movieId.toString());
  };

  // Sort movies: prioritize those with trailers
  const sortedMovieDetails = [...movieDetails].sort((a, b) => {
    const aHasTrailer = trailers[a.id];
    const bHasTrailer = trailers[b.id];
    if (aHasTrailer && !bHasTrailer) return -1;
    if (!aHasTrailer && bHasTrailer) return 1;
    return 0;
  });

  return (
    <div className="min-h-screen flex flex-col">
  <Navbar />
  <div className="container mx-auto text-white mt-40 mb-8 px-4">
        <h1 className="text-2xl mb-4">Find Similar Movies</h1>
        
        <div className="mb-6">
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Enter a movie title..."
              className="flex-grow p-2 rounded bg-gray-800 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button 
  type="submit"
  disabled={isLoading}
  className="px-4 py-2 bg-gradient-to-r font-bold from-blue-500 to-purple-500 rounded hover:from-blue-600 hover:to-purple-600 disabled:opacity-50 transition-all"
>
  {isLoading ? "Searching..." : "Search"}
</button>
          </form>
        </div>
        
        {error && (
          <div className="p-3 mb-4 bg-red-900/50 border border-red-700 rounded">
            {error}
          </div>
        )}
        
        {matchedTitle && (
          <div className="mb-4">
            <p className="text-gray-300">Closest Match: <span className="text-white font-semibold">{matchedTitle}</span></p>
          </div>
        )}
        
        {isLoading && (
          <div className="text-center py-8">
            <p>Searching for similar movies...</p>
          </div>
        )}
        
        {!isLoading && movieDetails.length > 0 && (
          <div>
            <h2 className="text-xl mb-3">Recommended Movies</h2>
            <div className="movie-cards grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
              {sortedMovieDetails
                .filter(movie => movie.poster_path) // Only include movies with poster images
                .map((movie) => {
                  const releaseYear = movie.release_date ? movie.release_date.split('-')[0] : 'Unknown';
                  const genres = movie.genres ? movie.genres.map(genre => genre.name).join(', ') : '';
                  const alreadyInWatchlist = isInWatchlist(movie.id);

                  return (
                    <div key={movie.id} className="movie-container flex flex-col">
                      <div className="movie-card relative group border border-[#1a2238] rounded-lg p-4 bg-gradient-to-b from-[#050810] to-[#100434] hover:bg-[#12172b] transition-all h-64 overflow-hidden">
                        <div className="flex h-full">
                          {/* Movie Image */}
                          <div className="movie-image-container h-full w-32 flex-shrink-0">
                            <img
                              src={`https://image.tmdb.org/t/p/w200${movie.poster_path}`}
                              alt={movie.title}
                              className="h-full w-full object-cover rounded"
                            />
                          </div>

                          {/* Movie Info Section */}
                          <div className="ml-4 flex flex-col justify-between flex-grow overflow-hidden">
                            <h3 className="text-xl font-semibold truncate max-w-full">{movie.title}</h3>
                            <p className="text-sm text-gray-400">{releaseYear}</p>
                            <p className="text-sm text-gray-400">{genres}</p>
                          </div>
                        </div>

                        {/* Trailer Overlay */}
                        <div 
                          className={`absolute top-0 left-0 w-full h-full bg-cover bg-center flex justify-center items-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-0 ${trailers[movie.id] ? '' : 'bg-black'}`}
                          style={{
                            backgroundImage: `url(https://image.tmdb.org/t/p/w500${movie.poster_path})`,
                          }}
                        >
                          {trailers[movie.id] && (
                            <a href={trailers[movie.id]} target="_blank" rel="noopener noreferrer">
                              <button className="bg-red-500 text-white py-2 px-4 rounded-lg">
                                Watch Trailer
                              </button>
                            </a>
                          )}
                        </div>
                      </div>
                      <div className="flex justify-between bg-black border border-gray-900 items-center space-x-4 relative z-10">
                        <button 
                          onClick={() => handleAddToWatchlist(movie)} 
                          className="bg-[#1d1158] bg-opacity-40 p-2 rounded-full hover:bg-opacity-70 transition-all"
                          title={alreadyInWatchlist ? "Already in watchlist" : "Add to Watchlist"}
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 5v14m7-7H5" />
                          </svg>
                        </button>
                        <button 
                          onClick={() => toggleDescription(movie.id)} 
                          className="bg-black bg-opacity-40 p-2 rounded-full hover:bg-opacity-70 transition-all"
                          title="Expand Description"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                          </svg>
                        </button>
                      </div>

                      {showDescription[movie.id] && movie.overview && (
                        <div className="mt-4 px-4 py-2 bg-gray-900 bg-opacity-50 rounded-md text-white">
                          <p>{movie.overview}</p>
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </div>
        )}
        
        {!isLoading && !error && movieDetails.length === 0 && searchTerm && recommendations.length === 0 && (
          <p className="text-gray-300">No recommendations found for this movie.</p>
        )}
      </div>
    </div>
  );
};

export default FindSimilarPage;