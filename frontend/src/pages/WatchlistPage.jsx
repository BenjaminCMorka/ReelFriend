import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from "../components/Navbar";
import { useAuthStore } from '../store/authStore';
import { toast } from 'react-hot-toast';



const WatchlistPage = () => {
  const [watchlistMovies, setWatchlistMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showDescription, setShowDescription] = useState({});
  const [trailers, setTrailers] = useState({});
  const [showRatingModal, setShowRatingModal] = useState(false);
  const [currentRating, setCurrentRating] = useState(0);
  const [currentMovieId, setCurrentMovieId] = useState(null);
  const [modalPosition, setModalPosition] = useState({ top: 0, left: 0 });
  const { user, isAuthenticated, removeFromWatchlist } = useAuthStore();
  const navigate = useNavigate();

  // Fetch movie details for each movie in the watchlist
  useEffect(() => {
    const fetchWatchlistMovies = async () => {
      if (!isAuthenticated || !user || !user.watchlist || user.watchlist.length === 0) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const apiKey = '7a0553e66258137e7f70085c7dde6cbc';
        const moviePromises = user.watchlist.map(async (movieId) => {
          const url = `https://api.themoviedb.org/3/movie/${movieId}?api_key=${apiKey}`;
          const response = await fetch(url);
          if (!response.ok) throw new Error(`Error fetching movie ${movieId}`);
          return response.json();
        });

        const moviesData = await Promise.all(moviePromises);
        setWatchlistMovies(moviesData);
        
        // Fetch trailers for each movie
        moviesData.forEach(movie => {
          fetchTrailer(movie.id);
        });
      } catch (err) {
        console.error('Error fetching watchlist movies:', err);
        setError('Failed to load your watchlist. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchWatchlistMovies();
  }, [isAuthenticated, user]);

  // Function to fetch trailers for each movie
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

  const toggleDescription = (movieId) => {
    setShowDescription(prevState => ({
      ...prevState,
      [movieId]: !prevState[movieId],
    }));
  };

  const handleRemoveFromWatchlist = async (movie) => {
    if (!isAuthenticated) {
      toast.error('Please login to manage your watchlist');
      navigate('/login');
      return;
    }
  
    try {
      const result = await removeFromWatchlist(
        movie.id.toString()
      );
      
      if (result.success === false) {
        toast.error(result.message || 'Failed to remove from watchlist');
      } else {
        toast.success(`Removed "${movie.title || movie.name}" from your watchlist!`);
        // Update local state to remove the movie from the UI immediately
        setWatchlistMovies(prevMovies => prevMovies.filter(m => m.id !== movie.id));
      }
    } catch (err) {
      console.error('Error removing from watchlist:', err);
      toast.error('Failed to remove from watchlist. Please try again.');
    }
  };

  const openRatingModal = (movieId, event) => {
    // Get the button's position to place the modal near it
    const buttonRect = event.currentTarget.getBoundingClientRect();
    setModalPosition({
      top: buttonRect.top + window.scrollY,
      left: buttonRect.left + window.scrollX
    });
    
    setCurrentMovieId(movieId);
    setCurrentRating(0);
    setShowRatingModal(true);
  };

  const handleRatingSubmit = async () => {
    try {
      // This would call an API to save the rating
      // For now, let's just show a success message
      const movieTitle = watchlistMovies.find(movie => movie.id === currentMovieId)?.title;
      toast.success(`You rated "${movieTitle}" ${currentRating} stars!`);
      
      // Remove from watchlist after rating
      setWatchlistMovies(prevMovies => prevMovies.filter(movie => movie.id !== currentMovieId));
      setShowRatingModal(false);
      
      // In a real implementation, you would call your backend API:
      // await axios.post('/api/movies/rate', { movieId: currentMovieId, rating: currentRating });
    } catch (err) {
      console.error('Error submitting rating:', err);
      toast.error('Failed to submit rating. Please try again.');
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navbar stays fixed at top */}
      <Navbar />
      
      {/* Main content */}
      <div className="container mx-auto text-white mt-25 mb-8 px-4">
        <h1 className="text-3xl font-bold mb-6 pt-4">Your Watchlist</h1>
        <hr className="border-t-2 border-white mb-6" />

        {loading && <p>Loading your watchlist...</p>}
        {error && <p className="text-red-500">{error}</p>}
        
        {!loading && watchlistMovies.length === 0 && (
          <div className="text-center py-12">
            <p className="text-xl mb-4">Your watchlist is empty</p>
            <button 
              onClick={() => navigate('/')} 
              className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded"
            >
              Discover Movies
            </button>
          </div>
        )}

        <div className="movie-cards grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {watchlistMovies.map((movie) => {
            const releaseYear = movie.release_date ? movie.release_date.split('-')[0] : 'Unknown';
            const genres = movie.genres ? movie.genres.map(genre => genre.name).join(', ') : '';

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
                  {/* Remove from watchlist button */}
                  <button 
                    onClick={() => handleRemoveFromWatchlist(movie)} 
                    className="bg-red-600 bg-opacity-40 p-2 rounded-full hover:bg-opacity-70 transition-all"
                    title="Remove from watchlist"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>

                  {/* Mark as watched button */}
                  <button 
                    onClick={(e) => openRatingModal(movie.id, e)} 
                    className="bg-[#2e2d2d] bg-opacity-40 p-2 rounded-full hover:bg-opacity-70 transition-all"
                    title="Mark as watched"
                  >
                    I've watched this
                  </button>

                  {/* Toggle description button */}
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

      {/* Rating Modal - Changed to be a popup instead of a full-screen overlay */}
      {showRatingModal && (
        <div className="fixed z-50" style={{ top: `${modalPosition.top}px`, left: `${modalPosition.left}px` }}>
          <div className="bg-[#1a2238] rounded-lg p-6 shadow-xl border border-purple-700 w-64">
            <h2 className="text-xl font-bold mb-4 text-white">
              Rate this movie
            </h2>
            
            <div className="flex justify-center space-x-2 my-6">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setCurrentRating(star)}
                  className={`text-3xl transition-colors ${
                    currentRating >= star ? 'text-yellow-400' : 'text-gray-400'
                  }`}
                >
                  ★
                </button>
              ))}
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowRatingModal(false)}
                className="px-4 py-2 border border-gray-500 text-white rounded hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleRatingSubmit}
                disabled={currentRating === 0}
                className={`px-4 py-2 rounded text-white ${
                  currentRating > 0 
                    ? 'bg-purple-600 hover:bg-purple-700' 
                    : 'bg-gray-500 cursor-not-allowed'
                }`}
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WatchlistPage;