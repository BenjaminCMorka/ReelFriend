import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const movies = [
  { title: "Inception", image: "/assets/inception.png" },
  { title: "Interstellar", image: "/assets/interstellar.jpg" },
  { title: "The Dark Knight", image: "/assets/dark-knight.jpeg" },
];

const LandingPage = () => {
    const [currentMovie, setCurrentMovie] = useState(movies[0]);
  
    useEffect(() => {
      const interval = setInterval(() => {
        setCurrentMovie(movies[Math.floor(Math.random() * movies.length)]);
      }, 4000);
      return () => clearInterval(interval);
    }, []);
  
    return (
      <div className="relative w-full h-screen overflow-hidden flex justify-center items-center">
        {/* Fullscreen Movie Background that changes every 4 seconds */}
        <div
          className="absolute inset-0 w-full h-full bg-cover bg-center transition-all duration-1000"
          style={{
            backgroundImage: `url(${currentMovie.image})`,
            filter: "brightness(50%)",
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        ></div>
  
        {/* Floating Particles */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/3 left-1/4 w-32 h-32 bg-purple-500 rounded-full opacity-20 blur-3xl"></div>
          <div className="absolute bottom-1/4 right-1/3 w-24 h-24 bg-blue-500 rounded-full opacity-20 blur-3xl"></div>
        </div>
  
        {/* Glassmorphism Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="relative z-10 flex flex-col items-center justify-center text-center text-white p-10 rounded-2xl bg-opacity-10 backdrop-blur-lg border border-white/20 shadow-lg mx-auto max-w-2xl"
        >
          <h1 className="text-6xl font-extrabold drop-shadow-lg bg-gradient-to-r from-blue-400 to-purple-700 text-transparent bg-clip-text">
            Welcome to Reel Friend
          </h1>
  
          <p className="text-2xl my-4 bg-gradient-to-r from-purple-400 to-blue-700 text-transparent bg-clip-text">
            Looking for something new to watch? Let's dive into a world of cinema
            and discover your new favorites. Ready to go?
          </p>
  
          {/* Buttons with Neon Glow */}
          <div className="flex justify-center space-x-6 mt-6">
            <motion.div whileHover={{ scale: 1.1 }}>
              <Link
                to="/signup"
                className="px-8 py-3 bg-purple-500 text-xl rounded-lg font-semibold transition-all shadow-lg 
                 hover:bg-blue-600 hover:shadow-blue-600/50"
              >
                Get Started
              </Link>
            </motion.div>
  
            <motion.div whileHover={{ scale: 1.1 }}>
              <Link
                to="/login"
                className="px-8 py-3 bg-blue-500 text-xl rounded-lg font-semibold transition-all shadow-lg hover:bg-blue-600 hover:shadow-blue-600/50"
              >
                Log In
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </div>
    );
  };
  
  export default LandingPage;
  