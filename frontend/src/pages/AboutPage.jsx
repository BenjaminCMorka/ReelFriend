import Navbar from "../components/Navbar";

function AboutPage() {
  return (
    <div className="min-h-screen text-white p-6 relative">
      <Navbar />
      <div className="max-w-3xl mx-auto space-y-12 pt-40">

        <div className="text-center">
          <h1 className="text-4xl font-bold pb-4">
            About ReelFriend
          </h1>
          <p className="text-lg text-gray-300 mt-6">
            ReelFriend is here to help you to discover movies you will love. 
            Whether you're in the mood for a thrilling adventure, an interesting drama, or a sci-fi, 
            we've got recommendations that feel like they come from a friend who really knows you.
          </p>
          <p className="text-lg text-gray-400 mt-4">
            Add movies to your watchlist, pass on the ones you're not interested in, and let ReelFriend refine 
            its recommendations for you. 
          </p>
        </div>


        <div className="w-full h-1 bg-gradient-to-r from-purple-500 to-blue-500"></div>


        <div className="text-center">
          <h2 className="text-2xl font-semibold pb-4">
            Explainability & Transparency
          </h2>
          <p className="text-lg text-gray-400 mt-6">
            ReelFriend isn’t a black box. Think of it like this, if a close friend recommended a movie to you, 
            they might explain why theyre recommending it. So, it's like they're giving you a personal explanation behind their choice, 
            not just saying, "I think you'd like this."
          </p>
          <p className="text-lg text-gray-400 mt-4">
            Just like that friend, ReelFriend provides insights into your recommendations. The more 
            you interact, the more personalized the recommendations become.
          </p>
        </div>
      </div>
    </div>
  );
}

export default AboutPage;
