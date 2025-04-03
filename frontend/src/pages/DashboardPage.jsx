import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import RecommendationCard from "../components/RecommendationCard";
import axios from "axios";
import { useAuthStore } from "../store/authStore";
import { toast } from "react-hot-toast";

const DashboardPage = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [movieDetails, setMovieDetails] = useState([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [trailers, setTrailers] = useState({});
  const [explanations, setExplanations] = useState([]);
  const [showRatingModal, setShowRatingModal] = useState(false);
  const [currentRating, setCurrentRating] = useState(0);
  const [currentMovieId, setCurrentMovieId] = useState(null);
  const [modalPosition, setModalPosition] = useState({ top: 0, left: 0 });

  const { 
    user, 
    isAuthenticated, 
    addToWatchlist, 
    isInWatchlist 
  } = useAuthStore();

  const fetchTrailer = async (movieId) => {
    const apiKey = "7a0553e66258137e7f70085c7dde6cbc";
    const url = `https://api.themoviedb.org/3/movie/${movieId}/videos?api_key=${apiKey}`;

    try {
      const response = await fetch(url);
      const data = await response.json();
      if (data.results && data.results.length > 0) {
        const trailer = data.results.find(video => video.type === "Trailer");
        if (trailer) {
          setTrailers(prevState => ({
            ...prevState,
            [movieId]: `https://www.youtube.com/watch?v=${trailer.key}`,
          }));
        }
      }
    } catch (err) {
      console.error("Error fetching trailer:", err);
    }
  };

  useEffect(() => {
    const getRecommendations = async () => {
      if (!isAuthenticated) {
        setIsLoading(false);
        setError("Please log in to view personalized recommendations");
        return;
      }
      try {
        setIsLoading(true);
        setError("");

        const response = await axios.post("http://localhost:5001/api/recommender", {}, {
          withCredentials: true
        });

        if (response.data.recommendations) {
          const validRecommendations = response.data.recommendations.filter(
            id => id && String(id).trim() !== ''
          );

          if (validRecommendations.length > 0) {
            setRecommendations(validRecommendations);
            setExplanations(response.data.explanations || []);
            setError("");
          } else {
            setError("Unable to find movie recommendations based on your preferences. Try adding more favorite movies.");
            setRecommendations([]);
            setExplanations([]);
          }
        } else if (response.data.message) {
          setError(response.data.message);
          setRecommendations([]);
          setExplanations([]);
        } else {
          setError("No recommendations available. Try updating your preferences.");
          setRecommendations([]);
          setExplanations([]);
        }
      } catch (err) {
        console.error("Error fetching dashboard recommendations:", err);
        setError(err.response?.data?.error || "Failed to get personalized recommendations");
        setRecommendations([]);
        setExplanations([]);
      } finally {
        setIsLoading(false);
      }
    };

    getRecommendations();
  }, [user, isAuthenticated]);

  useEffect(() => {
    const fetchMovieDetails = async (movieIds) => {
      if (!movieIds || movieIds.length === 0) return [];

      const apiKey = "7a0553e66258137e7f70085c7dde6cbc";
      const movieDetailsArray = [];
      
      // Get user's watchlist to filter recommendations
      const userWatchlist = user && user.watchlist ? user.watchlist : [];
      const watchedMovies = user && user.watchedMovies ? user.watchedMovies.map(movie => movie.movieId) : [];

      // Filter out movies already in watchlist or watched
      const filteredMovieIds = movieIds.filter(id => {
        return !userWatchlist.includes(id.toString()) && !watchedMovies.includes(id.toString());
      });

      for (const id of filteredMovieIds) {
        if (!id) continue;
        try {
          const response = await fetch(`https://api.themoviedb.org/3/movie/${id}?api_key=${apiKey}`);
          if (response.ok) {
            const data = await response.json();
            if (data) {
              movieDetailsArray.push(data);
              fetchTrailer(id);
            }
          } else {
            console.error(`Movie API returned ${response.status} for movie ID ${id}`);
          }
        } catch (err) {
          console.error(`Error fetching details for movie ${id}:`, err);
        }
      }

      return movieDetailsArray;
    };

    if (recommendations.length > 0) {
      const getMovieDetails = async () => {
        try {
          const details = await fetchMovieDetails(recommendations);
          setMovieDetails(details);
        } catch (err) {
          console.error("Error fetching movie details:", err);
          setMovieDetails([]);
        }
      };
      getMovieDetails();
    } else {
      setMovieDetails([]);
    }
  }, [recommendations, user]);

  const openRatingModal = (movieId, event) => {
    // Get the button's position to place the modal near it
    const buttonRect = event.currentTarget.getBoundingClientRect();
    setModalPosition({
      top: buttonRect.top + window.scrollY,
      left: buttonRect.left + window.scrollX
    });
    
    console.log("Opening rating modal for:", movieId);
    setCurrentMovieId(movieId);
    setCurrentRating(0);
    setShowRatingModal(true);
  };

  const handleRatingSubmit = async () => {
    if (!currentMovieId) return;

    try {
      const result = await useAuthStore.getState().markMovieAsWatched(currentMovieId, currentRating);
      if (result.success) {
        const movieTitle = movieDetails.find(movie => movie.id === currentMovieId)?.title;
        toast.success(`You rated "${movieTitle}" ${currentRating} stars!`);
        setShowRatingModal(false);
        setTimeout(() => {
          setMovieDetails(prev => prev.filter(movie => movie.id !== currentMovieId));
          setRecommendations(prev => prev.filter(id => id !== currentMovieId));
        }, 300);
      } else {
        toast.error(result.message || 'Failed to mark movie as watched');
      }
    } catch (err) {
      console.error('Full error in rating submission:', {
        errorMessage: err.message,
        errorResponse: err.response,
        errorStack: err.stack
      });
      const errorMessage = err.response?.data?.message || err.message || 'Failed to submit rating. Please try again.';
      toast.error(errorMessage);
    }
  };

  const sortedMovieDetails = [...movieDetails].sort((a, b) => {
    const aHasTrailer = trailers[a.id];
    const bHasTrailer = trailers[b.id];
    return aHasTrailer && !bHasTrailer ? -1 : !aHasTrailer && bHasTrailer ? 1 : 0;
  });

  const hasRecommendations = recommendations.length > 0 && movieDetails.length > 0;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }} className="min-h-screen flex flex-col">
      <Navbar />
      <div className="container mx-auto text-white mt-20 mb-8 px-4">
      {!isLoading && hasRecommendations && (
        <h2 className="text-xl font-medium mb-4 text-white">
          I found a few picks you might want to add to your list 
        </h2>
      )}
        <hr className="border-t border-gray-800 mb-6" />

        {error && (
          <div className="p-4 mb-6 bg-red-900/50 border border-red-700 rounded">
            {error}
          </div>
        )}

        {isLoading && (
          <div className="text-center py-8">
            <div className="animate-pulse flex flex-col items-center">
              <div className="h-8 w-64 bg-gray-700 rounded mb-4"></div>
              <div className="h-4 w-48 bg-gray-700 rounded"></div>
            </div>
            <p className="mt-4">Loading your personalized recommendations...</p>
          </div>
        )}

        {!isLoading && hasRecommendations && (
          <div>
            <h2 className="text-2xl font-semibold mb-4">Recommended For You</h2>
            <div className="movie-cards grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
              {sortedMovieDetails
                .filter(movie => movie && movie.poster_path)
                .map((movie) => {
                  const recIndex = recommendations.findIndex(rec => {
                    const recId = typeof rec === 'string' ? parseInt(rec, 10) : rec;
                    return !isNaN(recId) && recId === movie.id;
                  });

                  const explanation = recIndex >= 0 && recIndex < explanations.length 
                    ? explanations[recIndex] 
                    : "This movie matches your preferences.";

                  return (
                    <RecommendationCard
                      key={movie.id}
                      movie={movie}
                      explanation={explanation}
                      trailer={trailers[movie.id]}
                      isInWatchlist={isInWatchlist}
                      addToWatchlist={addToWatchlist}
                      openRatingModal={openRatingModal}
                    />
                  );
                })}
            </div>
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
    </motion.div>
  );
};
export default DashboardPage;