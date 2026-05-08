import { useState, useEffect } from 'react';
import { db, storage } from '../../lib/firebase';
import { collection, addDoc, getDocs, deleteDoc, doc, serverTimestamp, query, orderBy } from 'firebase/firestore';
import { ref, uploadBytesResumable, getDownloadURL, deleteObject } from 'firebase/storage';
import { useAuth } from '../../context/AuthContext';

export default function OfficerFiles() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const { isMember } = useAuth(); // or isAdmin depending on access level desired

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const q = query(collection(db, 'officer_files'), orderBy('createdAt', 'desc'));
      const querySnapshot = await getDocs(q);
      const filesList = querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      setFiles(filesList);
    } catch (err) {
      console.error("Error fetching files:", err);
      // It might fail if rules aren't set or index is missing. Let's fallback gracefully.
      try {
        const fallbackQ = query(collection(db, 'officer_files'));
        const fallbackSnapshot = await getDocs(fallbackQ);
        const fallbackList = fallbackSnapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data()
        }));
        setFiles(fallbackList);
      } catch (fallbackErr) {
        console.error("Fallback fetch also failed:", fallbackErr);
        setError('Failed to load files. Ensure Firestore is configured properly.');
      }
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check size limit (e.g., 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB.');
      return;
    }

    setUploading(true);
    setError('');
    
    // Create a storage ref
    const storageRef = ref(storage, `officer_files/${Date.now()}_${file.name}`);
    const uploadTask = uploadBytesResumable(storageRef, file);

    uploadTask.on(
      'state_changed',
      (snapshot) => {
        const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
        setUploadProgress(progress);
      },
      (err) => {
        console.error('Upload failed:', err);
        setError('Upload failed. Please try again.');
        setUploading(false);
      },
      async () => {
        try {
          const downloadURL = await getDownloadURL(uploadTask.snapshot.ref);
          
          // Save to Firestore
          await addDoc(collection(db, 'officer_files'), {
            name: file.name,
            size: file.size,
            type: file.type,
            url: downloadURL,
            storagePath: uploadTask.snapshot.ref.fullPath,
            createdAt: serverTimestamp(),
          });
          
          setUploading(false);
          setUploadProgress(0);
          e.target.value = ''; // clear input
          fetchFiles(); // refresh list
        } catch (dbErr) {
          console.error("Error saving to database:", dbErr);
          setError('File uploaded, but database record failed.');
          setUploading(false);
        }
      }
    );
  };

  const handleDelete = async (fileId, storagePath) => {
    if (!window.confirm("Are you sure you want to delete this file?")) return;
    
    try {
      // 1. Delete from Storage
      const fileRef = ref(storage, storagePath);
      await deleteObject(fileRef);
      
      // 2. Delete from Firestore
      await deleteDoc(doc(db, 'officer_files', fileId));
      
      // 3. Update UI
      setFiles(files.filter(f => f.id !== fileId));
    } catch (err) {
      console.error("Error deleting file:", err);
      setError('Failed to delete file. You may not have permission.');
    }
  };

  const formatSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Officer Files</h1>
        <p className="page-subtitle">Manage and share important documents for officers.</p>
      </div>

      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem', fontWeight: '600' }}>Upload New File</h2>
        {error && <div className="alert alert-error" style={{ marginBottom: '1rem', color: 'red' }}>{error}</div>}
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'flex-start' }}>
          <input 
            type="file" 
            id="file-upload"
            onChange={handleFileUpload}
            disabled={uploading}
            style={{ 
              padding: '0.5rem', 
              border: '1px solid var(--border-color)', 
              borderRadius: '6px',
              width: '100%',
              maxWidth: '400px'
            }} 
          />
          
          {uploading && (
            <div style={{ width: '100%', maxWidth: '400px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
                <span>Uploading...</span>
                <span>{Math.round(uploadProgress)}%</span>
              </div>
              <div style={{ height: '8px', background: 'var(--border-color)', borderRadius: '4px', overflow: 'hidden' }}>
                <div 
                  style={{ 
                    height: '100%', 
                    background: 'var(--primary-color)', 
                    width: `${uploadProgress}%`,
                    transition: 'width 0.3s ease'
                  }} 
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="files-grid" style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', 
        gap: '1.5rem' 
      }}>
        {files.length === 0 && !uploading && (
          <div className="empty-state" style={{ gridColumn: '1 / -1', padding: '3rem', textAlign: 'center', background: 'var(--card-bg)', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
            <p style={{ color: 'var(--text-secondary)' }}>No files have been uploaded yet.</p>
          </div>
        )}

        {files.map(file => (
          <div key={file.id} className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ flex: 1, marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                <div style={{ 
                  width: '40px', height: '40px', 
                  background: 'rgba(59, 130, 246, 0.1)', 
                  color: 'var(--primary-color)',
                  borderRadius: '8px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 'bold', fontSize: '1.2rem'
                }}>
                  📄
                </div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: '600', margin: 0, wordBreak: 'break-all' }}>
                  {file.name}
                </h3>
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', margin: 0 }}>
                {formatSize(file.size)} • {file.createdAt?.toDate ? file.createdAt.toDate().toLocaleDateString() : 'Just now'}
              </p>
            </div>
            
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
              <a 
                href={file.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn btn-primary"
                style={{ flex: 1, textAlign: 'center', textDecoration: 'none', padding: '0.5rem', borderRadius: '6px', fontSize: '0.875rem' }}
              >
                Download
              </a>
              <button 
                onClick={() => handleDelete(file.id, file.storagePath)}
                className="btn"
                style={{ flex: 1, padding: '0.5rem', borderRadius: '6px', fontSize: '0.875rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', border: 'none' }}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
