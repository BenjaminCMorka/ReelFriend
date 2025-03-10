import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuthStore } from "../store/authStore";
import { useNavigate } from 'react-router-dom';

import { Card, CardContent } from '../components/Card';
import Netflix from '../assets/Netflix_logo.png';
import Hulu from '../assets/Hulu_logo.png';
import Apple from '../assets/Apple_logo.png';
import Prime from '../assets/Prime_logo.png';
import Disney from '../assets/Disney_logo.png';
import Peacock from '../assets/Peacock_logo.png';
import Paramount from '../assets/Paramount_logo.png';
import HBO from '../assets/HBO_logo.png';

import SearchInput from '../components/SearchInput';

const OnboardingPage = () => {
    const { user, updateOnboarding, logout } = useAuthStore();
    const [showOnboarding, setShowOnboarding] = useState(!user?.hasOnboarded);
    const [currentStep, setCurrentStep] = useState(0);
    const [selectedGenres, setSelectedGenres] = useState([]);
    const [streamingServices, setStreamingServices] = useState([]);
    const [favMovies, setFavMovies] = useState([]);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    const [noResultsMessage, setNoResultsMessage] = useState('');
    const navigate = useNavigate();
    
    // Welcome animation states
    const [currentSlide, setCurrentSlide] = useState(0);
   
    
    const welcomeSlides = [
        `Hey, ${user?.name || "Benjamin"}`,
        "Now that we’re introduced...",
        "Let’s go find your next watch!"
    ];

    const questions = [
        'What Genres Do You Enjoy The Most?',
        'Tell me some movies you loved watching!',
        'What Streaming service/services do you use?'
    ];

    const genres = [
        'Action', 'Comedy', 'Drama', 'Horror', 'Romance',
        'Sci-Fi', 'Thriller', 'Adventure', 'Fantasy', 'Documentary', 'Mystery', 'Animation'
    ];

    const services = [
        { name: 'Netflix', logo: Netflix },
        { name: 'Hulu', logo: Hulu },
        { name: 'Amazon Prime', logo: Prime },
        { name: 'Disney+', logo: Disney },
        { name: 'HBO Max', logo: HBO },
        { name: 'Apple TV+', logo: Apple },
        { name: 'Peacock', logo: Peacock },
        { name: 'Paramount+', logo: Paramount }
    ];

    useEffect(() => {
        if (user) {
            setShowOnboarding(!user.hasOnboarded);
        }
    }, [user]);
    
    // Welcome animation effect
    useEffect(() => {
        if (!showOnboarding) {
            if (currentSlide < welcomeSlides.length - 1) {
                const timer = setTimeout(() => {
                    setCurrentSlide(prev => prev + 1);
                }, 1800); // Time each slide displays before transitioning
                
                return () => clearTimeout(timer);
            } 
        }
    }, [currentSlide, welcomeSlides.length, showOnboarding]);

    const toggleGenreSelection = (genre) => {
        setSelectedGenres(prev => prev.includes(genre) ? prev.filter(g => g !== genre) : [...prev, genre]);
    };

    const toggleServiceSelection = (service) => {
        setStreamingServices(prev => prev.includes(service) ? prev.filter(s => s !== service) : [...prev, service]);
    };

    const handleNext = async () => {
        console.log("Selected genres before sending:", selectedGenres);
        if (currentStep === questions.length - 1) {
            try {
                await updateOnboarding(selectedGenres, favMovies, streamingServices);
                setShowOnboarding(false);  // Close onboarding modal after successful update
                navigate('/dashboard');
            } catch (err) {
                console.error("Error updating onboarding data", err);
                setError("Error updating your preferences. Please try again.");
            }
        } else {
            setCurrentStep(prevStep => prevStep + 1);
        }
    };

    const handlePrev = () => {
        if (currentStep > 0) {
            setCurrentStep(prevStep => prevStep - 1);
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
            console.log("API Response Data:", data); // Check if the response data structure is correct
        
            if (data.results && data.results.length > 0) {
                setResults(data.results);
            } else {
                setNoResultsMessage(`No results found for "${searchTerm}".`);
            }
        } catch (err) {
            console.error('Error fetching movies:', err); // Make sure error is logged
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

    const handleLogout = () => {
        logout();  // Assuming logout is available in your authStore
    };

    return (
        <>
     
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ duration: 0.5 }}
                    className="max-w-md w-full mx-auto mt-10 p-8 bg-gray-900 bg-opacity-80 rounded-xl shadow-2xl text-center relative"
                >
                    {/* Onboarding Modal */}
                    <button
                        className="absolute top-4 right-4 text-white bg-red-600 rounded px-4 py-2"
                        onClick={handleLogout}
                    >
                        Logout
                    </button>

                    <div className="onboarding-modal bg-navy p-6 rounded-lg shadow-lg w-full mx-auto text-white">
                        <h2 className="text-xl font-bold">Let me get to know you better...</h2>
                        <p className="text-lg mt-2">{questions[currentStep]}</p>

                        {currentStep === 0 && (
                            <div className="grid grid-cols-3 gap-x-9 gap-y-4 my-5">
                                {genres.map(genre => (
                                    <button
                                        key={genre}
                                        className={`inline-flex items-center justify-center px-3 py-2 rounded-2xl text-white font-bold cursor-pointer text-sm transition-all
                                            ${selectedGenres.includes(genre)
                                                ? 'bg-[#29387a] border-2 border-[#e4910d]'
                                                : 'bg-[#29387a] border-2 border-transparent hover:bg-[#0e0e46]'
                                            } min-w-max whitespace-nowrap`}
                                        onClick={() => toggleGenreSelection(genre)}
                                    >
                                        {genre}
                                    </button>
                                ))}
                            </div>
                        )}

                        {currentStep === 1 && (
                            <div className="favorite-search">
                                <Card className="card">
                                    <CardContent className="card-content">
                                        <div className="search-container w-full">
                                            <SearchInput
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
                                    {results
                                        .filter(item => item.poster_path) // Only include items with a poster_path
                                        .map((item) => (
                                            <div key={item.id} className="movie-card">
                                                <img
                                                    src={`https://image.tmdb.org/t/p/w200${item.poster_path}`} // Use the poster path to display image
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
                                                            setFavMovies((prev) => [...prev, item.id]);
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

                        {currentStep === 2 && (
                            <div className="streaming-services-selection">
                                {services.map((service) => (
                                    <button
                                        key={service.name}
                                        className={`service-btn ${streamingServices.includes(service.name) ? 'selected' : ''}`}
                                        onClick={() => toggleServiceSelection(service.name)}
                                    >
                                        <img src={service.logo} alt={service.name} className="service-logo" />
                                    </button>
                                ))}
                            </div>
                        )}

                        <div className="action-buttons mt-6 flex justify-between">
                            <button className="prev-btn" onClick={handlePrev} disabled={currentStep === 0}>
                                Previous
                            </button>
                            <button className="next-btn" onClick={handleNext}>
                                {currentStep === questions.length - 1 ? 'Finish' : 'Next'}
                            </button>
                        </div>
                    </div>
                </motion.div>
            ) 
        </>
    );
};

export default OnboardingPage;