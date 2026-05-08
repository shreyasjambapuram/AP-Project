import { createContext, useContext, useState } from 'react';
import csvText from '../assets/S numbers - Science Olympiad.csv?raw';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [isAdmin, setIsAdmin] = useState(() => {
    return localStorage.getItem('scioly_admin') === 'true';
  });
  
  const [isMember, setIsMember] = useState(() => {
    return localStorage.getItem('scioly_member') === 'true' || localStorage.getItem('scioly_admin') === 'true';
  });

  const login = async (sNumber, passcode, role) => {
    // Validate S-Number (exactly 6 digits, optional leading 'S' or 's')
    const isValidSNumber = /^[sS]?\d{6}$/.test(sNumber);

    if (!isValidSNumber) {
      return { success: false, error: 'Invalid S-Number format. Must be 6 digits.' };
    }

    try {
      // Extract all 6 digit numbers from the imported CSV text, ignoring 's' prefixes to normalize
      const approvedNumbers = new Set(
        [...csvText.matchAll(/\b[sS]?(\d{6})\b/g)].map(match => match[1])
      );
      
      const normalizedInput = sNumber.toLowerCase().replace('s', '');
      
      if (!approvedNumbers.has(normalizedInput)) {
         return { success: false, error: 'Access Denied: S-Number not found in the approved list.' };
      }
    } catch (err) {
      console.error(err);
      return { success: false, error: 'Unable to verify S-Number at this time.' };
    }

    if (role === 'admin') {
      if (passcode === 'password') {
        setIsAdmin(true);
        setIsMember(true);
        localStorage.setItem('scioly_admin', 'true');
        localStorage.setItem('scioly_member', 'true');
        return { success: true };
      }
      return { success: false, error: 'Incorrect admin passcode.' };
    } else {
      setIsMember(true);
      localStorage.setItem('scioly_member', 'true');
      return { success: true };
    }
  };

  const logout = () => {
    setIsAdmin(false);
    setIsMember(false);
    localStorage.removeItem('scioly_admin');
    localStorage.removeItem('scioly_member');
  };

  return (
    <AuthContext.Provider value={{ isMember, isAdmin, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
