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

	// Fix the updateOnboarding function
	updateOnboarding: async (favoriteGenres, favoriteMovies, streamingServices) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axios.post(`${API_URL}/onboard`, { favoriteGenres, favoriteMovies,streamingServices });
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
		
		// Check if movie is already in watchlist
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
		
		// Check if the movie ID exists in the user's watchlist
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
	verifyEmail: async (code) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axios.post(`${API_URL}/verify-email`, { code });
			set({ user: response.data.user, isAuthenticated: true, isLoading: false });
			return response.data;
		} catch (error) {
			set({ error: error.response.data.message || "Error verifying email", isLoading: false });
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
	forgotPassword: async (email) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axios.post(`${API_URL}/forgot-password`, { email });
			set({ message: response.data.message, isLoading: false });
		} catch (error) {
			set({
				isLoading: false,
				error: error.response.data.message || "Error sending reset password email",
			});
			throw error;
		}
	},
	resetPassword: async (token, password) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axios.post(`${API_URL}/reset-password/${token}`, { password });
			set({ message: response.data.message, isLoading: false });
		} catch (error) {
			set({
				isLoading: false,
				error: error.response.data.message || "Error resetting password",
			});
			throw error;
		}
	},
}));