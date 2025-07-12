import bcryptjs from "bcryptjs";
import crypto from "crypto";

import { generateTokenAndSetCookie } from "../utils/generateTokenAndSetCookie.js";
import { User } from "../models/userModel.js";

export const signup = async (req, res) => {
	const { email, password, name } = req.body;

	try {
		if (!email || !password || !name) {
			throw new Error("All fields are required");
		}

		const userAlreadyExists = await User.findOne({ email });
		if (userAlreadyExists) {
			return res.status(400).json({ success: false, message: "User already exists" });
		}

		const hashedPassword = await bcryptjs.hash(password, 10);

		const user = new User({
			email,
			password: hashedPassword,
			name,
		});

		await user.save();
		generateTokenAndSetCookie(res, user._id);

		res.status(201).json({
			success: true,
			message: "User created successfully",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		res.status(400).json({ success: false, message: error.message });
	}
};

export const login = async (req, res) => {
	const { email, password } = req.body;
	try {
		const user = await User.findOne({ email });
		if (!user) {
			return res.status(400).json({ success: false, message: "Invalid credentials" });
		}
		const isPasswordValid = await bcryptjs.compare(password, user.password);
		if (!isPasswordValid) {
			return res.status(400).json({ success: false, message: "Invalid credentials" });
		}

		generateTokenAndSetCookie(res, user._id);

		await user.save();

		res.status(200).json({
			success: true,
			message: "Logged in successfully",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		console.log("Error in login ", error);
		res.status(400).json({ success: false, message: error.message });
	}
};

export const logout = async (req, res) => {
	res.clearCookie("token");
	res.status(200).json({ success: true, message: "Logged out successfully" });
};

export const checkAuth = async (req, res) => {
	try {
		const user = await User.findById(req.userId).select("-password");
		if (!user) {
			return res.status(400).json({ success: false, message: "User not found" });
		}

		res.status(200).json({ success: true, user });
	} catch (error) {
		console.log("Error in checkAuth ", error);
		res.status(400).json({ success: false, message: error.message });
	}
};

export const onboard = async (req, res) => {
    const { favoriteMovies } = req.body;
    try {
        const user = await User.findById(req.userId);
        if (!user) {
            return res.status(400).json({ success: false, message: "Invalid credentials" });
        }

        
        user.favoriteMovies = favoriteMovies.map(id => String(id));
        user.hasOnboarded = true;
        
       
        const watchedFromFavorites = favoriteMovies.map(movieId => ({
            movieId: String(movieId),
            rating: 4, // default rating for favorites
            watchedAt: new Date()
        }));

        user.watchedMovies = watchedFromFavorites;
        
        await user.save();
        generateTokenAndSetCookie(res, user._id);

        res.status(200).json({
            success: true,
            message: "Onboarding completed successfully",
            user: {
                ...user._doc,
                password: undefined,
            },
        });
    } catch (error) {
        console.error("Error in onboard:", error);
        res.status(400).json({ success: false, message: error.message });
    }
};

export const addToWatchlist = async (req, res) => {
	const { movieId, movieTitle, posterPath } = req.body;

	try {
		const user = await User.findById(req.userId);
		if (!user) {
			return res.status(400).json({ success: false, message: "User not found" });
		}

		if (user.watchlist.includes(movieId)) {
			return res.status(200).json({
				success: false,
				message: "Movie already in watchlist",
				user: {
					...user._doc,
					password: undefined,
				},
			});
		}

		user.watchlist.push(movieId);
		await user.save();

		res.status(200).json({
			success: true,
			message: "Movie added to watchlist",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		console.log("Error in addToWatchlist ", error);
		res.status(400).json({ success: false, message: error.message });
	}
};

export const removeFromWatchlist = async (req, res) => {
	const { movieId } = req.body;

	try {
		const user = await User.findById(req.userId);
		if (!user) {
			return res.status(400).json({ success: false, message: "User not found" });
		}

		const initialLength = user.watchlist.length;
		user.watchlist = user.watchlist.filter(id => id !== String(movieId));

		if (user.watchlist.length === initialLength) {
			return res.status(400).json({
				success: false,
				message: "Movie not found in watchlist",
			});
		}

		await user.save();

		res.status(200).json({
			success: true,
			message: "Movie removed from watchlist",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		console.log("Error in removeFromWatchlist ", error);
		res.status(400).json({ success: false, message: error.message });
	}
};

export const markMovieAsWatched = async (req, res) => {
	const { movieId, rating } = req.body;

	try {
		const user = await User.findById(req.userId);
		if (!user) {
			return res.status(400).json({ success: false, message: "User not found" });
		}

		if (!movieId) {
			return res.status(400).json({ success: false, message: "Movie ID is required" });
		}

		if (rating === undefined || rating === null) {
			return res.status(400).json({ success: false, message: "Rating is required" });
		}

		const existing = user.watchedMovies.find(m => m.movieId === String(movieId));

		if (existing) {
			existing.rating = rating;
			existing.watchedAt = new Date();
		} else {
			user.watchedMovies.push({
				movieId: String(movieId),
				rating,
				watchedAt: new Date(),
			});
		}

		if (user.modelTrainingStatus) {
			user.modelTrainingStatus.embeddingsUpdated = false;
		} else {
			user.modelTrainingStatus = {
				embeddingsUpdated: false,
			};
		}

		user.watchlist = user.watchlist.filter(id => id !== String(movieId));

		await user.save();

		res.status(200).json({
			success: true,
			message: "Movie marked as watched",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		console.error('Unexpected error in markMovieAsWatched:', error);
		res.status(500).json({
			success: false,
			message: "Unexpected error occurred",
		});
	}
};



export const resetPassword = async (req, res) => {
	const { newPassword } = req.body;

	try {
		const user = await User.findById(req.userId);
		if (!user) {
			return res.status(400).json({ success: false, message: "User not found" });
		}

		if (!newPassword) {
			return res.status(400).json({ success: false, message: "New password is required" });
		}

		const hashedPassword = await bcryptjs.hash(newPassword, 10);
		user.password = hashedPassword;

		await user.save();

		res.status(200).json({ success: true, message: "Password reset successfully" });
	} catch (error) {
		console.log("Error in resetPassword ", error);
		res.status(500).json({ success: false, message: error.message });
	}
};

export const updateProfile = async (req, res) => {
	const { name, email, currentPassword, newPassword } = req.body;

	try {
		const user = await User.findById(req.userId);
		if (!user) {
			return res.status(404).json({ success: false, message: "User not found" });
		}

		if (name) user.name = name;

		if (email && email !== user.email) {
			const exists = await User.findOne({ email });
			if (exists) {
				return res.status(400).json({ success: false, message: "Email already in use" });
			}
			user.email = email;
		}

		if (newPassword && currentPassword) {
			const isValid = await bcryptjs.compare(currentPassword, user.password);
			if (!isValid) {
				return res.status(400).json({ success: false, message: "Current password is incorrect" });
			}
			user.password = await bcryptjs.hash(newPassword, 10);
		}

		await user.save();

		res.status(200).json({
			success: true,
			message: "Profile updated successfully",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		console.error("Error in updateProfile:", error);
		res.status(500).json({ success: false, message: error.message || "Server error" });
	}
};
