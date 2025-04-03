import express from "express";
import { getRecommendations } from "../controllers/recommenderController.js";
import { verifyToken } from "../middleware/verifyToken.js";




const router = express.Router();

router.post("/", verifyToken, getRecommendations);


export default router;