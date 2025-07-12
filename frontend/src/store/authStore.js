import { create } from "zustand";
import axios from "axios";

const API_URL = "http://localhost:5001/api/auth";

axios.defaults.withCredentials = true;

export const useAuthStore = create((set) => ({
	user: null,
	isAuthenticated: false,

	error: null,
	isLoading: false,
	isCheckingAuth: true,
	hasWelcomed: false,
	message: null,
	recommendations: [],


	markMovieAsWatched: async (movieId, rating) => {
		set({ isLoading: true, error: null });
		
		try {
			console.log('Attempting to mark movie as watched:', { movieId, rating });
			
			const response = await axios.post(`${API_URL}/mark-watched`, { 
				movieId: String(movieId), 
				rating 
			}, {
				validateStatus: function (status) {
					return status >= 200 && status < 300; 
				}
			});
			
			console.log('Mark movie as watched response:', response);
			
			if (response.data.success) {
				set({
					user: response.data.user,
					error: null,
					isLoading: false
				});
				
				return response.data;
			} else {
				const errorMessage = response.data.message || "Failed to mark movie as watched";
				set({ 
					error: errorMessage, 
					isLoading: false 
				});
				throw new Error(errorMessage);
			}
		} catch (error) {
			console.error('Error in markMovieAsWatched:', {
				errorResponse: error.response,
				errorMessage: error.message,
				errorStack: error.stack
			});
			
			const errorMessage = error.response?.data?.message || 
								 error.message || 
								 "Error marking movie as watched";
			
			set({ 
				error: errorMessage, 
				isLoading: false 
			});
			
			throw new Error(errorMessage);
		}
	},

	updateOnboarding: async (favoriteMovies) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axios.post(`${API_URL}/onboard`, { favoriteMovies });
			set({
				isAuthenticated: true,
				user: response.data.user,
				error: null,
				isLoading: false,
			});
		} catch (error) {
			set({ error: error.response?.data?.message || "Error logging in", isLoading: false });
			throw error;
		}
	}, 
	addToWatchlist: async (movieId, movieTitle, posterPath) => {
		set({ isLoading: true, error: null });
		
		// check if movie is already in watchlist
		const { user } = useAuthStore.getState();
		if (user && user.watchlist && user.watchlist.includes(movieId)) {
		  set({ isLoading: false });
		  return { success: false, message: "Movie already in watchlist" };
		}
		
		try {
		  const response = await axios.post(`${API_URL}/watchlist/add`, { 
			movieId, 
			movieTitle, 
			posterPath 
		  });
		  set({
			user: response.data.user,
			error: null,
			isLoading: false,
		  });
		  return response.data;
		} catch (error) {
		  set({ 
			error: error.response?.data?.message || "Error adding to watchlist", 
			isLoading: false 
		  });
		  throw error;
		}
	  },

	  removeFromWatchlist: async (movieId, movieTitle, posterPath) => {
		set({ isLoading: true, error: null });
		
		try {
		  const response = await axios.post(`${API_URL}/watchlist/remove`, { 
			movieId, 
			movieTitle, 
			posterPath 
		  });
		  set({
			user: response.data.user,
			error: null,
			isLoading: false,
		  });
		  return response.data;
		} catch (error) {
		  set({ 
			error: error.response?.data?.message || "Error removing from watchlist", 
			isLoading: false 
		  });
		  throw error;
		}
	  },

	  isInWatchlist: (movieId) => {
		const { user } = useAuthStore.getState();
		if (!user || !user.watchlist) return false;
		
		// check if the movie ID exists in user's watchlist
		return user.watchlist.includes(movieId);
	  },
	


	

	signup: async (email, password, name) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axios.post(`${API_URL}/signup`, { email, password, name });
			set({ user: response.data.user, isAuthenticated: true, isLoading: false });
		} catch (error) {
			set({ error: error.response.data.message || "Error signing up", isLoading: false });
			throw error;
		}
	},
	login: async (email, password) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axios.post(`${API_URL}/login`, { email, password });
			set({
				isAuthenticated: true,
				user: response.data.user,
				error: null,
				isLoading: false,
			});
		} catch (error) {
			set({ error: error.response?.data?.message || "Error logging in", isLoading: false });
			throw error;
		}
	},

	logout: async () => {
		set({ isLoading: true, error: null });
		try {
			await axios.post(`${API_URL}/logout`);
			set({ user: null, isAuthenticated: false, error: null, isLoading: false });
		} catch (error) {
			set({ error: "Error logging out", isLoading: false });
			throw error;
		}
	},
	checkAuth: async () => {
		set({ isCheckingAuth: true, error: null });
		try {
			const response = await axios.get(`${API_URL}/check-auth`);
			set({ user: response.data.user, isAuthenticated: true, isCheckingAuth: false });
		} catch (error) {
			set({ error: null, isCheckingAuth: false, isAuthenticated: false });
            console.log(error);
		}
	},
	resetPassword: async (email, newPassword) => {
		set({ isLoading: true, error: null, message: null });
		try {
			const response = await axios.post(`${API_URL}/reset-password`, { email, newPassword });
			set({
				message: response.data.message || "Password reset successfully",
				isLoading: false,
				error: null,
			});
			return response.data;
		} catch (error) {
			set({
				error: error.response?.data?.message || "Error resetting password",
				isLoading: false,
			});
			throw error;
		}
	},
	
	
}));