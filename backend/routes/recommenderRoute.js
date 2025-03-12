import express from "express";
import { getRecommendations } from "../controllers/recommenderController.js";

const router = express.Router();

router.post("/", getRecommendations);

export default router;