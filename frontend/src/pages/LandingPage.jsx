
import { Link } from "react-router-dom";

const LandingPage = () => {
  return (
    <div>
      <div className="text-center">
        <h1 className="text-5xl font-bold mb-4 text-white">Welcome to Reel Friend</h1>
        <p className="mb-6 text-xl text-white">Your personalized movie recommendation assistant.</p>
        <div>
          <Link
            to="/signup"
            className="px-6 py-2 bg-purple-500 rounded-lg text-xl transition-all hover:bg-purple-600"
          >
            Sign Up
          </Link>
          <p className="my-4">or</p>
          <Link
            to="/login"
            className="px-6 py-2 bg-blue-500 rounded-lg text-xl transition-all hover:bg-blue-600"
          >
            Log In
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
