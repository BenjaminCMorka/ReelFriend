import { exec, execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

// Get the directory name in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const getRecommendations = async (req, res) => {
  const { movieTitle, count = 10 } = req.body;

  if (!movieTitle) {
    return res.status(400).json({ error: 'Movie title is required.' });
  }

  try {
    // Ensure the path is correct - we'll check that the file exists
    const scriptPath = path.resolve(__dirname, '../recommender/recommender.py');
    
    if (!fs.existsSync(scriptPath)) {
      console.error(`Python script not found at: ${scriptPath}`);
      return res.status(500).json({ 
        error: 'Recommender script not found. Check server configuration.' 
      });
    }
    
    console.log(`Executing Python script at: ${scriptPath}`);
    console.log(`Movie title: "${movieTitle}"`);

    // Use async exec instead of execSync for better performance
    exec(`python3 ${scriptPath} "${movieTitle}"`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Execution error: ${error.message}`);
        return res.status(500).json({ 
          error: 'Error executing the recommender script.',
          details: error.message 
        });
      }
      
      if (stderr) {
        console.error(`Python stderr: ${stderr}`);
      }
      
      console.log('Python script output:', stdout);
      
      // Try parsing the result as JSON
      try {
        const recommendations = JSON.parse(stdout);
        return res.json(recommendations);
      } catch (parseError) {
        console.error('Failed to parse Python output:', parseError);
        console.error('Raw output:', stdout);
        return res.status(500).json({ 
          error: 'Failed to parse the Python script output.',
          raw: stdout
        });
      }
    });
  } catch (error) {
    console.error('Error in recommendation controller:', error.message);
    return res.status(500).json({ 
      error: 'Failed to get recommendations. Please try again later.',
      details: error.message 
    });
  }
};