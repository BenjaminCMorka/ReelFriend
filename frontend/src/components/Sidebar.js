import React from 'react';
import Button from './ui/button';
import { useNavigate } from 'react-router-dom';
import { User, Home, TrendingUp, Eye, Search } from 'lucide-react';
import './styles/LandingPage.css'; 


const Sidebar = () => {
    const navigate = useNavigate();

    const goToHome = () => {
        navigate('/');
    };

    const goToTrending = () => {
        navigate('/trending');
    };

    const goToProfile = () => {
        navigate('/profile');
    };

    const goToWatchlist = () => {
        navigate('/watchlist');
    };

    const goToBrowse = () => {
        navigate('/browse');
    };


    return (
        <div className="sidebar">
            <div className="logo-container">
                <div className="logo">
                </div>
            </div>
            
            <nav className="nav">
            <Button className="nav-button" onClick={goToHome}>
                <Home className="icon" />
                Home
            </Button>
            <Button className="nav-button" onClick={goToWatchlist}>
                <Eye className="icon" />
                Watchlist
            </Button>
            <Button className="nav-button" onClick={goToTrending}>
                <TrendingUp className="icon" />
                Trending 
            </Button>
            <Button className="nav-button" onClick={goToBrowse}>
                <Search className="icon" />
                Browse
            </Button>
            <Button className="nav-button" onClick={goToProfile}>
                <User className="icon" />
                Profile
            </Button>
            </nav>
        </div>
    );
      
};
export default Sidebar;