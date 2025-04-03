import { useState } from "react";
import { toast } from "react-hot-toast";

const RecommendationCard = ({ 
  movie, 
  explanation,
  trailer,
  isInWatchlist,
  addToWatchlist,
  openRatingModal
}) => {
  const [showDescription, setShowDescription] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleAddToWatchlist = async () => {
    if (!movie || !movie.id) {
      toast.error("Movie information is not available");
      return;
    }

    if (isInWatchlist(movie.id.toString())) {
      toast.error(`"${movie.title}" is already in your watchlist!`);
      return;
    }

    try {
      await addToWatchlist(
        movie.id.toString(), 
        movie.title,
        movie.poster_path
      );
      toast.success(`Added "${movie.title}" to your watchlist!`);
    } catch (err) {
      console.error("Error adding to watchlist:", err);
      toast.error("Failed to add to watchlist. Please try again.");
    }
  };

  const toggleDescription = () => {
    setShowDescription(prev => !prev);
  };

  const toggleExplanation = () => {
    setShowExplanation(prev => !prev);
  };

  if (!movie) {
    return null;
  }

  const releaseYear = movie.release_date ? movie.release_date.split('-')[0] : 'Unknown';
  let genres = "";
  if (Array.isArray(movie.genres)) {
    genres = movie.genres.map(genre => genre.name).join(', ');
  } else {
    genres = movie.genres || 'Unknown';
  }

  return (
    <div className="movie-container flex flex-col">
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

        {/* Trailer Overlay */}
        <div 
          className={`absolute top-0 left-0 w-full h-full bg-cover bg-center flex justify-center items-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-0`}
          style={{
            backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url(https://image.tmdb.org/t/p/w500${movie.poster_path})`,
          }}
        >
          {trailer ? (
            <a href={trailer} target="_blank" rel="noopener noreferrer">
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
          onClick={handleAddToWatchlist} 
          className="text-white p-2 rounded hover:bg-gray-800 transition-colors"
          title="Add to Watchlist"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
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
            onClick={toggleDescription} 
            className="text-white p-2 rounded hover:bg-gray-800 transition-colors"
            title="Show Description"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </button>
          
          <button 
            onClick={toggleExplanation} 
            className="text-white p-2 rounded hover:bg-gray-800 transition-colors"
            title="Why Recommended"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
          </button>
        </div>
      </div>

      {showDescription && movie.overview && (
        <div className="mt-2 px-4 py-3 bg-gray-900 rounded-md text-white">
          <p className="text-sm">{movie.overview}</p>
        </div>
      )}


      {showExplanation && explanation && (
        <div className="mt-2 px-4 py-3 bg-blue-900/40 rounded-md text-white">
          <div className="flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-500 mt-0.5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <p className="text-sm flex-1">{explanation}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default RecommendationCard;