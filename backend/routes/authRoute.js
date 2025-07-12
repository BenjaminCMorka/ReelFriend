import express from "express";
import {
	login,
	logout,
	signup,
	resetPassword,
	checkAuth,
	onboard,
	addToWatchlist,
	removeFromWatchlist,
	markMovieAsWatched, updateProfile

} from "../controllers/authController.js";
import { verifyToken } from "../middleware/verifyToken.js";

const router = express.Router();

router.get("/check-auth", verifyToken, checkAuth);

router.post("/signup", signup);
router.post("/login", login);
router.post("/mark-watched", verifyToken, markMovieAsWatched);
router.post("/onboard", verifyToken, onboard);
router.post("/logout", logout);

router.put("/update-profile", verifyToken, updateProfile);
router.post("/reset-password", resetPassword);
router.post("/watchlist/add", verifyToken, addToWatchlist);
router.post("/watchlist/remove", verifyToken, removeFromWatchlist);

export default router;