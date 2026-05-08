import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const dbPath = join(__dirname, 'database.sqlite');
const db = new Database(dbPath);

const app = express();
app.use(cors());
app.use(express.json());

// Initialize table
db.exec(`
  CREATE TABLE IF NOT EXISTS store (
    collection TEXT,
    id TEXT,
    data TEXT,
    PRIMARY KEY (collection, id)
  )
`);

// Generic REST API

// Get all items in a collection
app.get('/api/:collection', (req, res) => {
  const { collection } = req.params;
  try {
    const rows = db.prepare('SELECT * FROM store WHERE collection = ?').all(collection);
    const items = rows.map(row => JSON.parse(row.data));
    res.json(items);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Initialize collection with sample data if empty
app.post('/api/:collection/init', (req, res) => {
  const { collection } = req.params;
  const sampleData = req.body;
  
  try {
    const count = db.prepare('SELECT count(*) as count FROM store WHERE collection = ?').get(collection).count;
    if (count === 0 && Array.isArray(sampleData)) {
      const insert = db.prepare('INSERT INTO store (collection, id, data) VALUES (?, ?, ?)');
      const insertMany = db.transaction((items) => {
        for (const item of items) {
          // ensure id exists
          if (!item.id) item.id = Date.now().toString() + Math.random().toString(36).substring(7);
          insert.run(collection, item.id, JSON.stringify(item));
        }
      });
      insertMany(sampleData);
    }
    
    // Return current state
    const rows = db.prepare('SELECT * FROM store WHERE collection = ?').all(collection);
    res.json(rows.map(row => JSON.parse(row.data)));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Add an item
app.post('/api/:collection', (req, res) => {
  const { collection } = req.params;
  const item = req.body;
  if (!item.id) {
    item.id = Date.now().toString();
  }
  
  try {
    db.prepare('INSERT INTO store (collection, id, data) VALUES (?, ?, ?)').run(
      collection, 
      item.id, 
      JSON.stringify(item)
    );
    res.json(item);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update an item
app.put('/api/:collection/:id', (req, res) => {
  const { collection, id } = req.params;
  const updates = req.body;
  
  try {
    const row = db.prepare('SELECT data FROM store WHERE collection = ? AND id = ?').get(collection, id);
    if (!row) {
      return res.status(404).json({ error: 'Item not found' });
    }
    
    const currentItem = JSON.parse(row.data);
    const updatedItem = { ...currentItem, ...updates, id }; // ensure ID doesn't change
    
    db.prepare('UPDATE store SET data = ? WHERE collection = ? AND id = ?').run(
      JSON.stringify(updatedItem),
      collection,
      id
    );
    res.json(updatedItem);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Delete an item
app.delete('/api/:collection/:id', (req, res) => {
  const { collection, id } = req.params;
  try {
    db.prepare('DELETE FROM store WHERE collection = ? AND id = ?').run(collection, id);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Backend Server running on http://localhost:${PORT}`);
});
