import { useState, useEffect } from 'react';
import { db, isConfigValid } from '../lib/firebase';
import { collection, getDocs, addDoc, updateDoc, deleteDoc, doc, setDoc } from 'firebase/firestore';
import { getData, setData, addItem, updateItem, removeItem } from '../utils/localStorage';

/**
 * A custom hook that transparently handles fetching and mutating data from
 * Firebase Firestore if configured, or falls back to localStorage if not.
 */
export function useDatabase(collectionName, sampleData = []) {
  const [data, setLocalData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Initialize and fetch data
  useEffect(() => {
    async function fetchData() {
      if (isConfigValid && db) {
        try {
          const querySnapshot = await getDocs(collection(db, collectionName));
          const items = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
          
          if (items.length === 0 && sampleData.length > 0) {
            // Seed database with sample data if empty
            const batchPromises = sampleData.map(async (item) => {
              const docRef = doc(collection(db, collectionName), item.id);
              await setDoc(docRef, item);
              return { ...item };
            });
            await Promise.all(batchPromises);
            setLocalData(sampleData);
          } else {
            setLocalData(items);
          }
        } catch (error) {
          console.error("Error fetching from Firebase:", error);
          fallbackToLocal();
        }
      } else {
        fallbackToLocal();
      }
      setLoading(false);
    }

    function fallbackToLocal() {
      let localItems = getData(collectionName);
      if (!localItems) {
        setData(collectionName, sampleData);
        localItems = sampleData;
      }
      setLocalData(localItems);
    }

    fetchData();
  }, [collectionName]);

  const add = async (item) => {
    if (isConfigValid && db) {
      try {
        const docRef = await addDoc(collection(db, collectionName), item);
        const newItem = { ...item, id: docRef.id };
        setLocalData(prev => [...prev, newItem]);
        return newItem;
      } catch (error) {
        console.error("Error adding document: ", error);
      }
    } else {
      const newItem = addItem(collectionName, item);
      setLocalData(prev => [...prev, newItem]);
      return newItem;
    }
  };

  const update = async (id, updates) => {
    if (isConfigValid && db) {
      try {
        const docRef = doc(db, collectionName, id);
        await updateDoc(docRef, updates);
        const updatedItem = { id, ...updates }; // Assuming updates contains all necessary fields here, or we fetch
        setLocalData(prev => prev.map(item => item.id === id ? { ...item, ...updates } : item));
        return updatedItem;
      } catch (error) {
        console.error("Error updating document: ", error);
      }
    } else {
      const updatedItem = updateItem(collectionName, id, updates);
      setLocalData(prev => prev.map(item => item.id === id ? updatedItem : item));
      return updatedItem;
    }
  };

  const remove = async (id) => {
    if (isConfigValid && db) {
      try {
        await deleteDoc(doc(db, collectionName, id));
        setLocalData(prev => prev.filter(item => item.id !== id));
      } catch (error) {
        console.error("Error deleting document: ", error);
      }
    } else {
      removeItem(collectionName, id);
      setLocalData(prev => prev.filter(item => item.id !== id));
    }
  };

  return { data, add, update, remove, loading };
}
