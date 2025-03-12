// components/MovieCard.jsx

import { useAuthStore } from '../store/authStore';
import { toast } from 'react-hot-toast';

const MovieCard = ({ movie, trailers, toggleDescription, handleAddToWatchlist }) => {
  const { user } = useAuthStore();
  const alreadyInWatchlist = user?.watchlist?.includes(movie.id.toString());
  const releaseYear = movie.release_date ? movie.release_date.split('-')[0] : 'Unknown';
  const genres = movie.genres ? movie.genres.map(genre => genre.name).join(', ') : '';

  return (
    <div className="movie-container flex flex-col">
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

      {/* Actions */}
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

      {movie.overview && (
        <div className="mt-4 px-4 py-2 bg-gray-900 bg-opacity-50 rounded-md text-white">
          <p>{movie.overview}</p>
        </div>
      )}
    </div>
  );
};

export default MovieCard;
