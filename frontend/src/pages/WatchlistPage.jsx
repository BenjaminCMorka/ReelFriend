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

  // fetch trailer for each movie if possible
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
        // remove movie from ui asap
        setWatchlistMovies(prevMovies => prevMovies.filter(m => m.id !== movie.id));
      }
    } catch (err) {
      console.error('Error removing from watchlist:', err);
      toast.error('Failed to remove from watchlist. Please try again.');
    }
  };

  const openRatingModal = (movieId, event) => {
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
        const movieTitle = watchlistMovies.find(movie => movie.id === currentMovieId)?.title;
        

        await useAuthStore.getState().markMovieAsWatched(currentMovieId, currentRating);
        
        toast.success(`You rated "${movieTitle}" ${currentRating} stars!`);
        
        // remove from watchlist after rating
        setWatchlistMovies(prevMovies => prevMovies.filter(movie => movie.id !== currentMovieId));
        setShowRatingModal(false);
    } catch (err) {
        console.error('Error submitting rating:', err);
        toast.error('Failed to submit rating. Please try again.');
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
    
      <Navbar />
      
     
      <div className="container mx-auto text-white mt-20 mb-8 px-4">
        <h1 className="text-3xl font-bold mb-6 pt-4">Your Watchlist</h1>
        <hr className="border-t border-gray-800 mb-6" />

        
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
            
                    <div className="movie-image-container h-full w-32 flex-shrink-0">
                      <img
                        src={`https://image.tmdb.org/t/p/w200${movie.poster_path}`}
                        alt={movie.title}
                        className="h-full w-full object-cover rounded"
                      />
                    </div>

  
                    <div className="ml-4 flex flex-col justify-between flex-grow overflow-hidden">
                      <h3 className="text-xl font-semibold truncate max-w-full">{movie.title}</h3>
                      <p className="text-sm text-gray-400">{releaseYear}</p>
                      <p className="text-sm text-gray-400 truncate">{genres}</p>
                    </div>
                  </div>

          
                  <div 
                    className="absolute top-0 left-0 w-full h-full bg-cover bg-center flex justify-center items-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-0"
                    style={{
                      backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url(https://image.tmdb.org/t/p/w500${movie.poster_path})`,
                    }}
                  >
                    {trailers[movie.id] ? (
                      <a href={trailers[movie.id]} target="_blank" rel="noopener noreferrer">
                        <button className="bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg shadow-lg transform hover:scale-105 transition-transform">
                          Watch Trailer
                        </button>
                      </a>
                    ) : (
                      <div className="text-white text-lg font-semibold">
                        No Trailer Available
                      </div>
                    )}
                  </div>
                </div>

                
                <div className="flex justify-between items-center bg-gray-900 border-t border-gray-800 p-2 relative z-10">
        
                <button 
                    onClick={() => handleRemoveFromWatchlist(movie)} 
                    className="text-white p-2 rounded hover:bg-gray-800 transition-colors"
                    title="Remove from watchlist"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                </button>


                  <div className="flex space-x-1">
            
                    <button 
                      onClick={(e) => openRatingModal(movie.id, e)} 
                      className="text-white p-2 rounded hover:bg-gray-800 transition-colors"
                      title="I've Watched This"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                      </svg>
                    </button>

   
                    <button 
                      onClick={() => toggleDescription(movie.id)} 
                      className="text-white p-2 rounded hover:bg-gray-800 transition-colors"
                      title="Show Description"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="6 9 12 15 18 9"></polyline>
                      </svg>
                    </button>
                  </div>
                </div>

                {showDescription[movie.id] && movie.overview && (
                  <div className="mt-2 px-4 py-3 bg-gray-900 rounded-md text-white">
                    <p className="text-sm">{movie.overview}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      
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