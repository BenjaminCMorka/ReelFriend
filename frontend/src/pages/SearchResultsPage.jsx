import { useParams } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Navbar from "../components/Navbar";
import { useAuthStore } from '../store/authStore';
import { toast } from 'react-hot-toast';
import RecommendationCard from "../components/RecommendationCard";
import { VALID_TMDB_IDS } from "../utils/tmdbIds";



const SearchResultsPage = () => {
  const { query } = useParams();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [noResultsMessage, setNoResultsMessage] = useState('');
  const [trailers, setTrailers] = useState({});
  const {  addToWatchlist, isInWatchlist } = useAuthStore();


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

  const [showRatingModal, setShowRatingModal] = useState(false);
  const [currentRating, setCurrentRating] = useState(0);
  const [currentMovieId, setCurrentMovieId] = useState(null);
  const [modalPosition, setModalPosition] = useState({ top: 0, left: 0 });

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
    if (!currentMovieId) return;

    try {
      const result = await useAuthStore.getState().markMovieAsWatched(currentMovieId, currentRating);
      if (result.success) {
        const movieTitle = results.find(movie => movie.id === currentMovieId)?.title;
        toast.success(`You rated "${movieTitle}" ${currentRating} stars!`);
        setShowRatingModal(false);
        

        setTimeout(() => {
          setResults(prev => prev.filter(movie => movie.id !== currentMovieId));
        }, 300);
      } else {
        toast.error(result.message || 'Failed to mark movie as watched');
      }
    } catch (err) {
      console.error('Error submitting rating:', err);
      toast.error(err.response?.data?.message || err.message || 'Failed to submit rating. Please try again.');
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
  
      try {
        // Fetch both search results and genre list concurrently
        const [searchRes, genreRes] = await Promise.all([
          fetch(`https://api.themoviedb.org/3/search/movie?api_key=${apiKey}&query=${query}`),
          fetch(`https://api.themoviedb.org/3/genre/movie/list?api_key=${apiKey}&language=en-US`)
        ]);
  
        const [searchData, genreData] = await Promise.all([
          searchRes.json(),
          genreRes.json()
        ]);
  
        const genreMap = {};
        genreData.genres.forEach(g => {
          genreMap[g.id] = g.name;
        });
  
        if (searchData.results && searchData.results.length > 0) {
          // Add genre names to each movie
          const enrichedResults = searchData.results.map(movie => ({
            ...movie,
            genres: movie.genre_ids.map(id => genreMap[id] || "Unknown")
          }));
  
          setResults(enrichedResults);
  
          enrichedResults.forEach(item => {
            if (item.id) fetchTrailer(item.id);
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
  

  // show movie with trailer first
  const sortedResults = [...results]
  .filter(movie => VALID_TMDB_IDS.has(String(movie.id))) 
  .sort((a, b) => {
    const aHasTrailer = trailers[a.id];
    const bHasTrailer = trailers[b.id];
    if (aHasTrailer && !bHasTrailer) return -1; 
    if (!aHasTrailer && bHasTrailer) return 1;   
    return 0;  
  });

  return (
    <div className="min-h-screen flex flex-col">

      <Navbar />
      
      <div className="container mx-auto text-white mt-20 mb-8 px-4">
        <h1 className="text-3xl font-bold mb-6 pt-4">
          {noResultsMessage ? `Search Results for "${query}"` : `Search Results for "${query}"`}
        </h1>
        <hr className="border-t border-gray-800 mb-6" />

        {loading && (
          <div className="text-center py-8">
            <div className="animate-pulse flex flex-col items-center">
              <div className="h-8 w-64 bg-gray-700 rounded mb-4"></div>
              <div className="h-4 w-48 bg-gray-700 rounded"></div>
            </div>
            <p className="mt-4">Searching for movies...</p>
          </div>
        )}
        
        {error && <p className="text-red-500 p-4 bg-red-900/50 border border-red-700 rounded">{error}</p>}
        
        {!loading && noResultsMessage && (
          <p className="text-center py-8">{noResultsMessage}</p>
        )}

        {!loading && results.length > 0 && (
          <div className="movie-cards grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
            {sortedResults
              .filter(item => item.poster_path) // only include movies with a poster
              .map((movie) => (
                <RecommendationCard
  key={movie.id}
  movie={movie}
  explanation={`This movie matched your search for "${query}".`}
  trailer={trailers[movie.id]}
  isInWatchlist={isInWatchlist}
  addToWatchlist={addToWatchlist}
  openRatingModal={openRatingModal}
/>

              ))}
          </div>
        )}
      </div>


      {showRatingModal && (
        <div className="fixed z-50" style={{ top: `${modalPosition.top}px`, left: `${modalPosition.left}px` }}>
          <div className="bg-[#1a2238] rounded-lg p-6 shadow-xl border border-purple-700 w-64">
            <h2 className="text-xl font-bold mb-4 text-white">Rate this movie</h2>
            <div className="flex justify-center space-x-2 my-6">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setCurrentRating(star)}
                  className={`text-3xl transition-colors ${currentRating >= star ? 'text-yellow-400' : 'text-gray-400'}`}
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
                className={`px-4 py-2 rounded text-white ${currentRating > 0 ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-500 cursor-not-allowed'}`}
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

export default SearchResultsPage;