import express from "express";

import {
    getRecommendations,
} from "../controllers/recommenderController.js";

const router = express.Router();
router.post("/recommender", getRecommendations);

export default router;