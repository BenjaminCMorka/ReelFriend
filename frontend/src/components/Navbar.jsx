import { Link, useLocation } from "react-router-dom";
import { FaUserCircle } from "react-icons/fa";
import { useAuthStore } from "../store/authStore";
import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';

const Navbar = () => {
  const location = useLocation(); 
  const { logout } = useAuthStore();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [scrolled, setScrolled] = useState(false); 
  const handleLogout = () => {
    logout();
  };

  const handleSearch = () => {
    if (query) {
      navigate(`/search/${encodeURIComponent(query)}`);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();  
    }
  };


  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setScrolled(true); 
      } else {
        setScrolled(false);
      }
    };

   
    window.addEventListener('scroll', handleScroll);


    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <nav 
      className={`text-white py-4 px-6 flex items-center justify-between w-full fixed top-0 left-0 z-50 transition-all ${scrolled ? 'bg-black bg-opacity-80' : 'bg-transparent'}`}
    >

      <Link
        to="/"
        className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-700 text-transparent bg-clip-text hover:opacity-80 transition-opacity"
      >
        ReelFriend
      </Link>

 
      <div className="absolute left-1/2 transform -translate-x-1/2 space-x-6 flex">
        <NavItem to="/dashboard" currentPath={location.pathname}>
          Dashboard
        </NavItem>
        <Separator />
        <NavItem to="/watchlist" currentPath={location.pathname}>
          My Watchlist
        </NavItem>
        <Separator />
        <NavItem to="/about" currentPath={location.pathname}>
          About
        </NavItem>
      </div>


      <div className="flex items-center space-x-4">

        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}  
          placeholder="Search movies..."
          className="px-4 py-2 rounded-full text-white bg-gray-900 bg-opacity-10 placeholder-opacity-70 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-all"
        />


        <div className="relative group">
          <button
            className="text-white text-2xl hover:opacity-80"
            onMouseEnter={(e) => e.target.classList.add("opacity-80")}
            onMouseLeave={(e) => e.target.classList.remove("opacity-80")}
          >
            <FaUserCircle />
          </button>

 
          <div className="absolute top-10 right-0 bg-gray-800 text-white py-2 px-4 rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <Link
              to="/account"
              className="block hover:bg-gray-700 py-1 px-2 rounded-md"
            >
              Account
            </Link>
            <button
              onClick={handleLogout}
              className="block hover:bg-gray-700 py-1 px-2 rounded-md w-full text-left"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

const NavItem = ({ to, currentPath, children }) => {
  const isActive = currentPath === to;

  return (
    <Link to={to} className="relative hover:text-gray-400 pb-2">
      {children}
      {isActive && (
        <span className="absolute left-0 bottom-0 w-full h-0.5 bg-gradient-to-r from-blue-400 to-purple-600"></span>
      )}
    </Link>
  );
};

const Separator = () => {
  return (
    <span className="text-white opacity-60">|</span>
  );
};

export default Navbar;
