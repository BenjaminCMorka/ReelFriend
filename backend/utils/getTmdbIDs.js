import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const csvPath = path.join(__dirname, '../recommender/data/links.csv');
const csvContent = fs.readFileSync(csvPath, 'utf8');
const lines = csvContent.split('\n');

// find the tmdbId column index
const headers = lines[0].split(',');
const tmdbIdIndex = headers.findIndex(h => h.trim().toLowerCase() === 'tmdbid');

// extract the IDs
const tmdbIds = new Set();
for (let i = 1; i < lines.length; i++) {
  const line = lines[i].trim();
  if (!line) continue;

  const columns = line.split(',');
  if (columns.length > tmdbIdIndex) {
    const tmdbId = columns[tmdbIdIndex].trim();
    if (tmdbId && tmdbId !== 'null' && tmdbId !== 'undefined' && tmdbId !== '') {
      tmdbIds.add(tmdbId);
    }
  }
}

// generate the code for the Set
const idsCode = `export const VALID_TMDB_IDS = new Set([\n  ${Array.from(tmdbIds).map(id => `"${id}"`).join(', ')}\n]);\n`;

// write to a file
const outputPath = path.join(__dirname, 'tmdbIds.js');
fs.writeFileSync(outputPath, idsCode);

