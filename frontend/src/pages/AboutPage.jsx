import Navbar from "../components/Navbar";

function AboutPage() {
  return (
    <div className="min-h-screen text-white p-6 relative">
      <Navbar />
      <div className="max-w-3xl mx-auto space-y-12 pt-40">
        {/* About Section */}
        <div className="text-center">
          <h1 className="text-4xl font-bold pb-4">
            About ReelFriend
          </h1>
          <p className="text-lg text-gray-300 mt-6">
            ReelFriend is your personal movie companion, helping you discover movies tailored to your taste. 
            Whether you're in the mood for a thrilling adventure, a heartwarming drama, or a mind-bending sci-fi, 
            we've got recommendations that feel like they come from a friend who truly knows you.
          </p>
          <p className="text-lg text-gray-400 mt-4">
            Add movies to your watchlist, pass on the ones you're not interested in, and let ReelFriend refine 
            its recommendations just for you. No distractions—just great movies, picked for you.
          </p>
        </div>

        {/* Gradient Separator */}
        <div className="w-full h-1 bg-gradient-to-r from-purple-500 to-blue-500"></div>

        {/* Explainability Section */}
        <div className="text-center">
          <h2 className="text-2xl font-semibold pb-4">
            Explainability & Transparency
          </h2>
          <p className="text-lg text-gray-400 mt-6">
            ReelFriend isn’t a black box. Think of it like this—if a close friend recommended a movie to you, 
            they might say something like, "I know you love action-packed movies with unexpected twists, 
            so I thought you'd enjoy this one!" Or, "You told me how much you liked 'Inception,' and this movie 
            has the same mind-bending vibe." It's like they're giving you a personal explanation behind their choice, 
            not just saying, "I think you'd like this."
          </p>
          <p className="text-lg text-gray-400 mt-4">
            Just like that friend, ReelFriend provides insights into your matches—whether it's based on your 
            favorite genres, past likes, or even preferences shared by other users with similar tastes. The more 
            you interact, the more personalized and understandable the recommendations become.
          </p>
        </div>
      </div>
    </div>
  );
}

export default AboutPage;
