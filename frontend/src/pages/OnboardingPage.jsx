import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuthStore } from "../store/authStore";
import { useNavigate } from 'react-router-dom';

import { Card, CardContent } from '../components/Card';
import SearchInput from '../components/SearchInput';

const OnboardingPage = () => {
    const { user, updateOnboarding, logout } = useAuthStore();
    const [favMovies, setFavMovies] = useState([]);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    const [noResultsMessage, setNoResultsMessage] = useState('');
    const navigate = useNavigate();

    const question = 'Tell me some movies you loved watching!';

    useEffect(() => {
        if (user && user.hasOnboarded) {
            navigate('/dashboard');
        }
    }, [user, navigate]);

    const handleSearch = async () => {
        setLoading(true);
        setError('');
        setNoResultsMessage('');
        setResults([]);

        if (!searchTerm.trim()) {
            setNoResultsMessage("Please enter a movie title.");
            setLoading(false);
            return;
        }

        const apiKey = '7a0553e66258137e7f70085c7dde6cbc';
        const url = `https://api.themoviedb.org/3/search/multi?api_key=${apiKey}&query=${searchTerm}`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();

            if (data.results && data.results.length > 0) {
                setResults(data.results);
            } else {
                setNoResultsMessage(`No results found for "${searchTerm}".`);
            }
        } catch (err) {
            console.error('Error fetching movies:', err);
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

    const handleSubmit = async () => {
        try {
            await updateOnboarding([], favMovies, []);
            navigate('/dashboard');
        } catch (err) {
            console.error("Error updating onboarding data", err);
            setError("Failed to save your preferences. Please try again.");
        }
    };

    const handleLogout = () => {
        logout();
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
            className="max-w-md w-full mx-auto mt-10 p-8 bg-gray-900 bg-opacity-80 rounded-xl shadow-2xl text-center relative"
        >
            <button
                className="absolute top-4 right-4 text-white bg-red-600 rounded px-4 py-2"
                onClick={handleLogout}
            >
                Logout
            </button>

            <div className="text-white">
                <h2 className="text-xl font-bold">Let me get to know you better...</h2>
                <p className="text-lg mt-2">{question}</p>

                <Card className="card mt-4">
                    <CardContent className="card-content">
                        <SearchInput
                            type="text"
                            placeholder="Search for Movies or Series..."
                            className="input"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            onKeyDown={handleKeyDown}
                        />
                    </CardContent>
                </Card>

                {loading && <p>Loading...</p>}
                {error && <p className="text-red-500">{error}</p>}
                {noResultsMessage && <p className="text-yellow-500">{noResultsMessage}</p>}

                <div className="selected-movies my-4">
                    {favMovies.map((movie, index) => (
                        <div key={index} className="selected-movie-box flex justify-between items-center bg-gray-800 p-2 rounded my-2">
                            <span>{movie.title || movie.name}</span>
                            <button
                                onClick={() => setFavMovies((prev) => prev.filter((_, i) => i !== index))}
                                className="text-red-500 text-sm"
                            >
                                ✖
                            </button>
                        </div>
                    ))}
                </div>

                <div className="movie-cards grid grid-cols-2 gap-4 my-4">
                    {results
                        .filter(item => item.poster_path)
                        .map((item) => (
                            <div key={item.id} className="movie-card bg-gray-800 p-2 rounded">
                                <img
                                    src={`https://image.tmdb.org/t/p/w200${item.poster_path}`}
                                    alt={item.title || item.name}
                                    className="movie-image rounded mb-2"
                                />
                                <h3 className="text-sm font-semibold text-white">{item.title || item.name}</h3>
                                <button
                                    className="mt-2 text-sm text-blue-400 hover:underline"
                                    onClick={() => {
                                        setFavMovies((prev) => [...prev, item]);
                                        setSearchTerm('');
                                        setResults([]);
                                    }}
                                >
                                    Select
                                </button>
                            </div>
                        ))}
                </div>

                <button
                    className="mt-6 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                    onClick={handleSubmit}
                    disabled={favMovies.length < 3}
                >
                    Finish
                </button>
            </div>
        </motion.div>
    );
};

export default OnboardingPage;
