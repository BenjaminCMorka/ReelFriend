import { useParams, useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Navbar from "../components/Navbar";
import { useAuthStore } from '../store/authStore';
import { toast } from 'react-hot-toast';

const genreNames = {
  28: 'Action',
  12: 'Adventure',
  16: 'Animation',
  35: 'Comedy',
  80: 'Crime',
  99: 'Documentary',
  18: 'Drama',
  10751: 'Family',
  14: 'Fantasy',
  36: 'History',
  27: 'Horror',
  10402: 'Music',
  9648: 'Mystery',
  10749: 'Romance',
  878: 'Science Fiction',
  10770: 'TV Movie',
  53: 'Thriller',
  10752: 'War',
  37: 'Western',
};

const SearchResultsPage = () => {
  const { query } = useParams(); // Capture the dynamic query from the URL
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [noResultsMessage, setNoResultsMessage] = useState('');
  const [trailers, setTrailers] = useState({});
  const [showDescription, setShowDescription] = useState({});
  const { user, isAuthenticated, addToWatchlist } = useAuthStore();
  const navigate = useNavigate();

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

  

  useEffect(() => {
    const handleSearch = async () => {
      setLoading(true);
      setError('');
      setNoResultsMessage('');

      if (!query.trim()) {
        setNoResultsMessage("Please Enter Movie Title.");
        setLoading(false);
        return;
      }

      const apiKey = '7a0553e66258137e7f70085c7dde6cbc';
      const url = `https://api.themoviedb.org/3/search/movie?api_key=${apiKey}&query=${query}`;

      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        console.log("API Response Data:", data);

        if (data.results && data.results.length > 0) {
          setResults(data.results);
          data.results.forEach(item => {
            if (item.id) {
              fetchTrailer(item.id); // Fetch trailer for each movie
            }
          });
        } else {
          setResults([]); 
          setNoResultsMessage(`Oops, I couldn't find anything for "${query}".`);
        }
      } catch (err) {
        console.error('Error fetching movies:', err);
        setError('Failed to fetch movies. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    handleSearch();
  }, [query]);

  const toggleDescription = (movieId) => {
    setShowDescription(prevState => ({
      ...prevState,
      [movieId]: !prevState[movieId],
    }));
  };

  const handleAddToWatchlist = async (movie) => {
    if (!isAuthenticated) {
      // Redirect to login if user is not authenticated
      toast.error('Please login to add movies to your watchlist');
      navigate('/login');
      return;
    }

    // Check if movie is already in watchlist
    if (user?.watchlist?.includes(movie.id.toString())) {
      toast.error(`"${movie.title || movie.name}" is already in your watchlist!`);
      return;
    }

    try {
      const result = await addToWatchlist(
        movie.id.toString(), 
        movie.title || movie.name,
        movie.poster_path
      );
      
      if (result.success === false && result.message === "Movie already in watchlist") {
        toast.error(`"${movie.title || movie.name}" is already in your watchlist!`);
      } else {
        toast.success(`Added "${movie.title || movie.name}" to your watchlist!`);
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
  const sortedResults = results.sort((a, b) => {
    const aHasTrailer = trailers[a.id];
    const bHasTrailer = trailers[b.id];
    if (aHasTrailer && !bHasTrailer) return -1;  // a comes first if it has a trailer
    if (!aHasTrailer && bHasTrailer) return 1;   // b comes first if it has a trailer
    return 0;  // If both or neither have trailers, leave their order as is
  });

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navbar stays fixed at top */}
      <Navbar />
      
      {/* Main content with fixed top spacing and aligned to top */}
      <div className="container mx-auto text-white mt-25 mb-8 px-4">
        <h1 className="text-3xl font-bold mb-6 pt-4">
          {noResultsMessage ? `Oops, I couldn't find anything for "${query}"` : `Here's what I found for "${query}"`}
        </h1>
        <hr className="border-t-2 border-white mb-6" />

        {loading && <p>Loading...</p>}
        {error && <p className="text-red-500">{error}</p>}
        {noResultsMessage && <p>{noResultsMessage}</p>}

        <div className="movie-cards grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {sortedResults
            .filter(item => item.poster_path) // Only include items with a poster_path
            .map((item) => {
              const releaseYear = item.release_date ? item.release_date.split('-')[0] : item.first_air_date ? item.first_air_date.split('-')[0] : 'Unknown';
              const genres = item.genre_ids.map(genreId => genreNames[genreId] || 'Unknown').join(', ');
              const alreadyInWatchlist = isInWatchlist(item.id);

              return (
                <div key={item.id} className="movie-container flex flex-col">
                 <div className="movie-card relative group border border-[#1a2238] rounded-lg p-4 bg-gradient-to-b from-[#050810] to-[#100434]
 hover:bg-[#12172b] transition-all h-64 overflow-hidden">

                    <div className="flex h-full">
                      {/* Movie Image */}
                      <div className="movie-image-container h-full w-32 flex-shrink-0">
                        <img
                          src={`https://image.tmdb.org/t/p/w200${item.poster_path}`}
                          alt={item.title || item.name}
                          className="h-full w-full object-cover rounded"
                        />
                      </div>

                      {/* Movie Info Section */}
                      <div className="ml-4 flex flex-col justify-between flex-grow overflow-hidden">
                        <h3 className="text-xl font-semibold truncate max-w-full">{item.title || item.name}</h3>
                        <p className="text-sm text-gray-400">{releaseYear}</p>
                        <p className="text-sm text-gray-400">{genres}</p>
                      </div>
                    </div>

                    {/* Trailer Overlay */}
                    <div 
                      className={`absolute top-0 left-0 w-full h-full bg-cover bg-center flex justify-center items-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-0 ${trailers[item.id] ? '' : 'bg-black'}`}
                      style={{
                        backgroundImage: `url(https://image.tmdb.org/t/p/w500${item.poster_path})`,
                      }}
                    >
                      {trailers[item.id] && (
                        <a href={trailers[item.id]} target="_blank" rel="noopener noreferrer">
                          <button className="bg-red-500 text-white py-2 px-4 rounded-lg">
                            Watch Trailer
                          </button>
                        </a>
                      )}
                    </div>
                  </div>
                  <div className="flex justify-between bg-black border  border-gray-900 items-center space-x-4 relative z-10">
                    <button 
                      onClick={() => handleAddToWatchlist(item)} 
                      className="bg-[#1d1158] bg-opacity-40 p-2 rounded-full hover:bg-opacity-70 transition-all"
                      title={alreadyInWatchlist ? "Already in watchlist" : "Add to Watchlist"}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 5v14m7-7H5" />
                      </svg>
                    </button>
                    <button 
                      onClick={() => toggleDescription(item.id)} 
                      className="bg-black bg-opacity-40 p-2 rounded-full hover:bg-opacity-70 transition-all"
                      title="Expand Description"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                  </div>

                  {showDescription[item.id] && item.overview && (
                    <div className="mt-4 px-4 py-2 bg-gray-900 bg-opacity-50 rounded-md text-white">
                      <p>{item.overview}</p>
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};

export default SearchResultsPage;