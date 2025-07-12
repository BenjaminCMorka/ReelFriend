import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuthStore } from "../store/authStore";
import { useNavigate } from "react-router-dom";
import { VALID_TMDB_IDS } from "../utils/tmdbIds";

const OnboardingPage = () => {
	const { user, updateOnboarding, logout } = useAuthStore();
	const [favMovies, setFavMovies] = useState([]);
	const [searchTerm, setSearchTerm] = useState("");
	const [results, setResults] = useState([]);
	const [error, setError] = useState("");
	const [loading, setLoading] = useState(false);
	const navigate = useNavigate();

	useEffect(() => {
		if (user?.hasOnboarded) navigate("/dashboard");
	}, [user, navigate]);

	const handleSearch = async () => {
		if (!searchTerm.trim()) return;

		setLoading(true);
		setError("");
		setResults([]);

		const apiKey = "7a0553e66258137e7f70085c7dde6cbc";
		const url = `https://api.themoviedb.org/3/search/movie?api_key=${apiKey}&query=${searchTerm}`;

		try {
			const response = await fetch(url);
			const data = await response.json();
			setResults(data.results || []);
		} catch (err) {
			console.error("Search error:", err);
			setError("Could not fetch movies. Please try again.");
		} finally {
			setLoading(false);
		}
	};

	const handleSubmit = async () => {
		try {
            const movieIds = favMovies.map(movie => movie.id.toString());
			await updateOnboarding(movieIds);
			navigate("/dashboard");
		} catch (err) {
			console.error("Submit error:", err);
			setError("Failed to save preferences.");
		}
	};

	const handleLogout = () => logout();

	return (
		<div className="min-h-screen text-white px-4 py-6">

			<div className="absolute top-4 left-6 text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-700 text-transparent bg-clip-text select-none">
				ReelFriend
			</div>

			<div className="absolute top-4 right-6">
				<button
					onClick={handleLogout}
					className="text-sm px-5 py-2.5 rounded-md font-medium bg-neutral-800 hover:bg-neutral-700 transition"
				>
					Logout
				</button>
			</div>

	
			<motion.div
				initial={{ opacity: 0, y: 30 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.4 }}
				className="w-full max-w-4xl mx-auto bg-gray-900 rounded-xl border border-neutral-800 shadow-md mt-50 p-15"
			>
				<h1 className="text-2xl font-semibold mb-4 text-center">
					What movies have you loved?
				</h1>
				<p className="text-sm text-white text-center mb-6">
					Search and add at least 3 favorites to help me personalize your recommendations.
				</p>

				{/* Search */}
				<div className="flex gap-2 mb-6">
                    <input
                        type="text"
                        placeholder="Search for a movie..."
                        value={searchTerm}
                        onChange={(e) => {
                            const value = e.target.value;
                            setSearchTerm(value);

                            if (value.trim() === "") {
                                setResults([]);
                                setError("");
                            }
                        }}
                        onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                        className="w-full px-4 py-2 rounded bg-gray-900 border border-neutral-800 text-sm focus:outline-none focus:ring-1 focus:ring-neutral-700"
                    />

					<button
						onClick={handleSearch}
						className="px-4 py-2 text-sm rounded border bg-blue-900 border-neutral-700 hover:bg-purple-900 transition"
					>
						Search
					</button>
				</div>


				{favMovies.length > 0 && (
					<div className="mb-6">
						<h3 className="text-sm mb-2 text-neutral-400">Selected Movies:</h3>
						<ul className="space-y-2">
							{favMovies.map((movie, idx) => (
								<li
                                key={idx}
                                className="flex justify-between items-center bg-gray-800 px-4 py-2 rounded"
                            >
                                <span className="text-sm">
                                    {movie.title}{" "}
                                    <span className="text-neutral-400">
                                        ({movie.release_date ? new Date(movie.release_date).getFullYear() : "N/A"})
                                    </span>
                                </span>
                                <button
                                    onClick={() =>
                                        setFavMovies((prev) => prev.filter((_, i) => i !== idx))
                                    }
                                    className="text-red-400 hover:text-red-900 text-xs"
                                >
                                    Remove
                                </button>
                            </li>
                            
							))}
						</ul>
					</div>
				)}

			
				{loading && <p className="text-sm text-neutral-500">Searching...</p>}
				{error && <p className="text-sm text-red-500">{error}</p>}

				{results.length > 0 && (
					<div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
						{results
  .filter(r => r.poster_path && VALID_TMDB_IDS.has(String(r.id)))
  .map((movie) => (
								<div
	key={movie.id}
	className="border border-gray-700 rounded-lg p-2 text-center hover:bg-gray-700 transition"
>
	<img
		src={`https://image.tmdb.org/t/p/w200${movie.poster_path}`}
		alt={movie.title}
		className="rounded mb-2"
	/>
	<p className="text-xs mb-2">
		{movie.title}{" "}
		<span className="text-neutral-400">
			({movie.release_date ? new Date(movie.release_date).getFullYear() : "N/A"})
		</span>
	</p>
	<button
		onClick={() => {
			setFavMovies((prev) => [...prev, movie]);
			setSearchTerm("");
			setResults([]);
		}}
		className="w-7 h-7 flex items-center justify-center rounded-full bg-green-600 text-white hover:bg-green-700 transition mx-auto"
		aria-label="Add movie"
	>
		+
	</button>
</div>


							))}
					</div>
				)}

				<button
					onClick={handleSubmit}
					disabled={favMovies.length < 3}
					className="w-full py-3 text-sm font-medium rounded bg-gradient-to-r from-blue-400 to-purple-700 text-white shadow hover:opacity-90 transition disabled:opacity-50"
				>
					Finish Onboarding
				</button>
			</motion.div>
		</div>
	);
};

export default OnboardingPage;
