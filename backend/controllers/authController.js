import bcryptjs from "bcryptjs";
import crypto from "crypto";

import { generateTokenAndSetCookie } from "../utils/generateTokenAndSetCookie.js";
import {
	sendPasswordResetEmail,
	sendResetSuccessEmail,
	sendVerificationEmail,
} from "../mailtrap/emails.js";
import { User } from "../models/userModel.js";



export const signup = async (req, res) => {
	const { email, password, name } = req.body;

	try {
		if (!email || !password || !name) {
			throw new Error("All fields are required");
		}

		const userAlreadyExists = await User.findOne({ email });
		console.log("userAlreadyExists", userAlreadyExists);

		if (userAlreadyExists) {
			return res.status(400).json({ success: false, message: "User already exists" });
		}

		const hashedPassword = await bcryptjs.hash(password, 10);
		const verificationToken = Math.floor(100000 + Math.random() * 900000).toString();

		const user = new User({
			email,
			password: hashedPassword,
			name,
			verificationToken,
			verificationTokenExpiresAt: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
		});

		await user.save();


		generateTokenAndSetCookie(res, user._id);

		await sendVerificationEmail(user.email, verificationToken);

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

export const addToWatchlist = async (req, res) => {
	const { movieId, movieTitle, posterPath } = req.body;
	
	try {
	  const user = await User.findById(req.userId);
	  if (!user) {
		return res.status(400).json({ success: false, message: "User not found" });
	  }
	  
	  // check if movie is already in watchlist
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
	  
	  // add movie to watchlist
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
	  
	  // remove movie from watchlist
	  const initialWatchlistLength = user.watchlist.length;
	  user.watchlist = user.watchlist.filter(id => id !== String(movieId));
	  
	  // check if the movie was actually removed
	  if (user.watchlist.length === initialWatchlistLength) {
		return res.status(400).json({ 
		  success: false, 
		  message: "Movie not found in watchlist" 
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
export const verifyEmail = async (req, res) => {
	const { code } = req.body;
	try {
		const user = await User.findOne({
			verificationToken: code,
			verificationTokenExpiresAt: { $gt: Date.now() },
		});

		if (!user) {
			return res.status(400).json({ success: false, message: "Invalid or expired verification code" });
		}

		user.isVerified = true;
		user.verificationToken = undefined;
		user.verificationTokenExpiresAt = undefined;
		await user.save();


		res.status(200).json({
			success: true,
			message: "Email verified successfully",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		console.log("error in verifyEmail ", error);
		res.status(500).json({ success: false, message: "Server error" });
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
export const onboard = async (req, res) => {
	const { favoriteMovies } = req.body;

	try {
		const user = await User.findById(req.userId);
		if (!user) {
			return res.status(400).json({ success: false, message: "Invalid credentials" });
		}

		// update user with favoriteMovies and set onboarding flag
		user.favoriteMovies = favoriteMovies;
		user.hasOnboarded = true;

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



export const logout = async (req, res) => {
	res.clearCookie("token");
	res.status(200).json({ success: true, message: "Logged out successfully" });
};

export const forgotPassword = async (req, res) => {
	const { email } = req.body;
	try {
		const user = await User.findOne({ email });

		if (!user) {
			return res.status(400).json({ success: false, message: "User not found" });
		}

		// generate reset token
		const resetToken = crypto.randomBytes(20).toString("hex");
		const resetTokenExpiresAt = Date.now() + 1 * 60 * 60 * 1000; // 1 hour

		user.resetPasswordToken = resetToken;
		user.resetPasswordExpiresAt = resetTokenExpiresAt;

		await user.save();

		// send email
		await sendPasswordResetEmail(user.email, `${process.env.CLIENT_URL}/reset-password/${resetToken}`);

		res.status(200).json({ success: true, message: "Password reset link sent to your email" });
	} catch (error) {
		console.log("Error in forgotPassword ", error);
		res.status(400).json({ success: false, message: error.message });
	}
};
export const updateProfile = async (req, res) => {
	const { name, email, currentPassword, newPassword } = req.body;
	
	try {
	  const user = await User.findById(req.userId);
	  if (!user) {
		return res.status(404).json({ success: false, message: "User not found" });
	  }
	  
	  // update name if provided
	  if (name) {
		user.name = name;
	  }
	  
	  // update email if provided
	  if (email && email !== user.email) {
		const existingUser = await User.findOne({ email });
		if (existingUser) {
		  return res.status(400).json({ success: false, message: "Email already in use" });
		}
		user.email = email;
	  }
	  
	  // update password if provided
	  if (newPassword && currentPassword) {
		const isPasswordValid = await bcryptjs.compare(currentPassword, user.password);
		if (!isPasswordValid) {
		  return res.status(400).json({ success: false, message: "Current password is incorrect" });
		}
		
		const hashedPassword = await bcryptjs.hash(newPassword, 10);
		user.password = hashedPassword;
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

export const resetPassword = async (req, res) => {
	try {
		const { token } = req.params;
		const { password } = req.body;

		const user = await User.findOne({
			resetPasswordToken: token,
			resetPasswordExpiresAt: { $gt: Date.now() },
		});

		if (!user) {
			return res.status(400).json({ success: false, message: "Invalid or expired reset token" });
		}

		// update password
		const hashedPassword = await bcryptjs.hash(password, 10);

		user.password = hashedPassword;
		user.resetPasswordToken = undefined;
		user.resetPasswordExpiresAt = undefined;
		await user.save();

		await sendResetSuccessEmail(user.email);

		res.status(200).json({ success: true, message: "Password reset successful" });
	} catch (error) {
		console.log("Error in resetPassword ", error);
		res.status(400).json({ success: false, message: error.message });
	}
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

export const markMovieAsWatched = async (req, res) => {
    const { movieId, rating } = req.body;
    
    console.log('Received mark movie as watched request:', { 
        userId: req.userId, 
        movieId, 
        rating 
    });
    
    try {
        const user = await User.findById(req.userId);
        if (!user) {
            console.error('User not found for ID:', req.userId);
            return res.status(400).json({ 
                success: false, 
                message: "User not found" 
            });
        }
        
        // Validate inputs
        if (!movieId) {
            console.error('Missing movie ID');
            return res.status(400).json({ 
                success: false, 
                message: "Movie ID is required" 
            });
        }
        
        if (rating === undefined || rating === null) {
            console.error('Missing or invalid rating');
            return res.status(400).json({ 
                success: false, 
                message: "Rating is required" 
            });
        }
        
        // check if the movie is already in watched movies
        const existingWatchedMovie = user.watchedMovies.find(
            movie => movie.movieId === String(movieId)
        );
        
        if (existingWatchedMovie) {
            // update existing rating
            existingWatchedMovie.rating = rating;
            existingWatchedMovie.watchedAt = new Date();
        } else {
            // add new watched movie
            user.watchedMovies.push({
                movieId: String(movieId),
                rating: rating,
                watchedAt: new Date()
            });
        }
        
        // flag that embeddings need to be updated
        if (user.modelTrainingStatus) {
            user.modelTrainingStatus.embeddingsUpdated = false;
        } else {
            user.modelTrainingStatus = {
                embeddingsUpdated: false
            };
        }
        
        // remove from watchlist if it exists
        user.watchlist = user.watchlist.filter(id => id !== String(movieId));
        
        try {
            await user.save();
            
            console.log('Successfully marked movie as watched');
            
            res.status(200).json({
                success: true,
                message: "Movie marked as watched",
                user: {
                    ...user._doc,
                    password: undefined,
                },
            });
        } catch (saveError) {
            console.error('Error saving user document:', saveError);
            res.status(500).json({ 
                success: false, 
                message: "Error saving user data" 
            });
        }
    } catch (error) {
        console.error('Unexpected error in markMovieAsWatched:', error);
        res.status(500).json({ 
            success: false, 
            message: "Unexpected error occurred" 
        });
    }
};