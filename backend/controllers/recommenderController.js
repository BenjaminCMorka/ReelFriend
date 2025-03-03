import { execSync } from 'child_process';
import path from 'path';

export const getRecommendations = async (req, res) => {
  const { movieTitle, count } = req.body;

  if (!movieTitle || !count) {
    return res.status(400).json({ error: 'Movie title and count are required.' });
  }

  try {
    // Ensure the path is correct, and use path.resolve to make it absolute
    const scriptPath = path.resolve(__dirname, '../recommender/content.py');

    // Execute the Python script with input parameters
    const result = execSync(`python3 ${scriptPath} "${movieTitle}" ${count}`).toString();

    // Try parsing the result as JSON
    let recommendations;
    try {
      recommendations = JSON.parse(result);
    } catch (parseError) {
      return res.status(500).json({ error: 'Failed to parse the Python script output.' });
    }

    // Send the parsed recommendations back as a response
    return res.json(recommendations);
  } catch (error) {
    console.error('Error executing Python script:', error);
    return res.status(500).json({ error: 'Failed to get recommendations. Please try again later.' });
  }
};
