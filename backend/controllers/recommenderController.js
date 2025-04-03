import { exec } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { User } from "../models/userModel.js";
import { Recommendation } from "../models/recommendationModel.js";
import mongoose from 'mongoose';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function executeRecommendationScript(userId, favoriteMovies, mode = 'load', newRatings = null) {
  const scriptPath = path.resolve(__dirname, '../recommender/main.py');
  
  if (!fs.existsSync(scriptPath)) {
      console.error(`Python script not found at: ${scriptPath}`);
      throw new Error('Recommender script not found.');
  }
  
  const favoriteMoviesStr = favoriteMovies.map(id => String(id)).join(',');
  
  console.log(`Passing favorite movies to Python script: ${favoriteMoviesStr}`);
  
  let command = `cd ${path.dirname(scriptPath)} && python3 ${scriptPath} --data-path "data" --mode ${mode} --favorite-movies "${favoriteMoviesStr}" --user-id ${userId}`;
  
  // Add new ratings parameter if provided
  if (newRatings && mode === 'update_embedding') {
      // Escape quotes in JSON string to prevent command line issues
      const escapedRatings = newRatings.replace(/"/g, '\\"');
      command += ` --new-ratings "${escapedRatings}"`;
  }
  
  console.log(`Executing command: ${command}`);
  
  return new Promise((resolve, reject) => {
      exec(command, { maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
          if (error && !stdout && !fs.existsSync(path.resolve(path.dirname(scriptPath), 'recommendation_results.json'))) {
              console.error(`Execution error: ${error.message}`);
              reject(error);
              return;
          }
          
          console.log(`Python stdout length: ${stdout.length} characters`);
          if (stderr) {
              console.error(`Python stderr: ${stderr}`);
          }
          
          const startMarker = "RESULTS_JSON_START";
          const endMarker = "RESULTS_JSON_END";
          
          const startIndex = stdout.indexOf(startMarker);
          const endIndex = stdout.indexOf(endMarker);
          
          let resultsData = null;
          
          if (startIndex !== -1 && endIndex !== -1) {
              const jsonStr = stdout.substring(startIndex + startMarker.length, endIndex).trim();
              
              try {
                  resultsData = JSON.parse(jsonStr);
                  console.log("Successfully parsed results from stdout");
              } catch (parseError) {
                  console.error("Error parsing JSON from stdout:", parseError);
              }
          }
          
          if (!resultsData) {
              const resultsPath = path.resolve(path.dirname(scriptPath), 'recommendation_results.json');
              
              if (fs.existsSync(resultsPath)) {
                  const fileData = fs.readFileSync(resultsPath, 'utf8');
                  
                  try {
                      resultsData = JSON.parse(fileData);
                      console.log("Successfully parsed results from file");
                  } catch (fileParseError) {
                      console.error("Error parsing JSON from file:", fileParseError);
                      throw new Error("Failed to parse recommendation results");
                  }
              } else {
                  console.error("Results file not found and no output from script");
                  throw new Error("No recommendation results available");
              }
          }
          
          if (resultsData.recommendations) {
              const validRecommendations = resultsData.recommendations.filter(
                  id => id && String(id).trim() !== ''
              );
              
              if (validRecommendations.length > 0) {
                  resolve({
                      recommendations: validRecommendations,
                      explanations: resultsData.explanations || []
                  });
              } else {
                  resolve({
                      recommendations: [
                          "299534", "299536", "24428", "299537", "10138", 
                          "76341", "100402", "284053", "118340", "245891"
                      ],
                      explanations: Array(10).fill("Recommended based on your favorite movies.")
                  });
              }
          } else {
              resolve({
                  recommendations: [
                      "299534", "299536", "24428", "299537", "10138", 
                      "76341", "100402", "284053", "118340", "245891"
                  ],
                  explanations: Array(10).fill("Recommended based on your favorite movies.")
              });
          }
      });
  });
}

export const getRecommendations = async (req, res) => {
  const userId = req.userId;

  try {
      const user = await User.findById(userId);
      if (!user) {
          return res.status(404).json({ error: 'User not found' });
      }

      let favoriteMovies = user.favoriteMovies || [];

      const highlyRatedWatchedMovies = user.watchedMovies
          .filter(movie => movie.rating >= 3)
          .map(movie => movie.movieId);

      favoriteMovies = [...new Set([...favoriteMovies, ...highlyRatedWatchedMovies])];

      if (favoriteMovies.length === 0) {
          return res.json({
              recommendations: [],
              explanations: [],
              message: "Please add some favorite movies to get personalized recommendations."
          });
      }

      const existingRecommendation = await Recommendation.findOne({ 
          user: userId,
          expiresAt: { $gt: new Date() }
      });

      // Enhanced decision logic
      const MIN_RATED_MOVIES = 5;
      let shouldRetrain = false;
      let shouldUpdateEmbedding = false;
      
      // If this is a new user who hasn't been trained yet
      if (!user.modelTrainingStatus?.lastTrainedAt) {
          // New user needs full training once they've rated enough movies
          shouldRetrain = user.watchedMovies.filter(movie => movie.rating > 0).length >= MIN_RATED_MOVIES;
          console.log(`New user detection: Rated movies count = ${user.watchedMovies.filter(movie => movie.rating > 0).length}, Should retrain = ${shouldRetrain}`);
      } else if (user.modelTrainingStatus?.embeddingsUpdated === false) {
          // Existing user with the embeddingsUpdated flag set to false
          shouldUpdateEmbedding = true;
          console.log(`User ${userId} has new ratings, will update embeddings`);
      }
      
      // If cached recommendations exist and no updates needed, return cached
      if (existingRecommendation && !shouldRetrain && !shouldUpdateEmbedding) {
          return res.json({
              recommendations: existingRecommendation.movieIds,
              explanations: existingRecommendation.explanations,
              cached: true
          });
      }

      // Generate recommendations, potentially retraining or updating embeddings
      const mode = shouldRetrain ? 'train' : (shouldUpdateEmbedding ? 'update_embedding' : 'load');
      const recommendationResult = await executeRecommendationScript(userId, favoriteMovies, mode);

      // Upsert recommendations
      await Recommendation.findOneAndUpdate(
          { user: userId }, 
          {
              user: userId,
              movieIds: recommendationResult.recommendations,
              explanations: recommendationResult.explanations,
              generatedAt: new Date(),
              expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000)
          },
          { 
              upsert: true,
              new: true,
              setDefaultsOnInsert: true
          }
      );

      // Update model training status
      if (!user.modelTrainingStatus) {
          user.modelTrainingStatus = {};
      }
      
      if (shouldRetrain) {
          user.modelTrainingStatus.lastTrainedAt = new Date();
          user.modelTrainingStatus.ratedMoviesCount = user.watchedMovies.filter(movie => movie.rating > 0).length;
          user.modelTrainingStatus.embeddingsUpdated = true;
      } else if (shouldUpdateEmbedding) {
          // Mark embeddings as updated
          user.modelTrainingStatus.embeddingsUpdated = true;
      }
      
      await user.save();

      return res.json({
          recommendations: recommendationResult.recommendations,
          explanations: recommendationResult.explanations,
          cached: false,
          retrained: shouldRetrain,
          embeddingUpdated: shouldUpdateEmbedding
      });
      
  } catch (error) {
      console.error('Error in dashboard recommendations:', error);
      
      return res.json({
          recommendations: [
              "299534", "299536", "24428", "299537", "10138", 
              "76341", "100402", "284053", "118340", "245891"
          ],
          explanations: Array(10).fill("Recommended based on your favorite movies."),
          error: error.message || 'Failed to get personalized recommendations.'
      });
  }
};