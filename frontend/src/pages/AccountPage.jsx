import { useState, useEffect } from "react";
import { toast } from "react-hot-toast";
import axios from "axios";
import Navbar from "../components/Navbar";
import { useAuthStore } from "../store/authStore";
import { motion } from "framer-motion";

const AccountPage = () => {
  const { user, isAuthenticated, checkAuth } = useAuthStore();
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    currentPassword: "",
    newPassword: "",
    confirmNewPassword: "",
  });

  useEffect(() => {
    if (user) {
      setFormData(prevData => ({
        ...prevData,
        name: user.name || "",
        email: user.email || "",
        currentPassword: "",
        newPassword: "",
        confirmNewPassword: "",
      }));
    }
  }, [user]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    

    if (formData.newPassword !== formData.confirmNewPassword) {
      toast.error("New password and confirmation do not match");
      return;
    }

    if (formData.newPassword && !formData.currentPassword) {
      toast.error("Current password is required to set a new password");
      return;
    }

    try {
      setIsLoading(true);
      
      // Only include the fields that have changed
      const updatedFields = {};
      
      if (formData.name !== user.name) {
        updatedFields.name = formData.name;
      }
      
      if (formData.newPassword) {
        updatedFields.currentPassword = formData.currentPassword;
        updatedFields.newPassword = formData.newPassword;
      }
      
      // make the api call only when theres changes
      if (Object.keys(updatedFields).length > 0) {
        const response = await axios.put(
          "http://localhost:5001/api/auth/update-profile",
          updatedFields,
          { withCredentials: true }
        );
        
        if (response.data.success) {
          toast.success("Profile updated successfully!");
          
   
          setFormData(prevData => ({
            ...prevData,
            currentPassword: "",
            newPassword: "",
            confirmNewPassword: "",
          }));
          
       
          await checkAuth();
        }
      } else {
        toast.info("No changes to save");
      }
    } catch (error) {
      const errorMessage = error.response?.data?.message || "Failed to update profile";
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <div className="container mx-auto mt-24 px-4 text-center text-white">
          <h2 className="text-2xl font-semibold mb-4">Please log in to view your account</h2>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      transition={{ duration: 0.5 }} 
      className="min-h-screen flex flex-col"
    >
      <Navbar />
      
      <div className="container mx-auto mt-24 px-4 text-white">
        <h1 className="text-3xl font-bold mb-6">Account Settings</h1>
        <hr className="border-t border-gray-800 mb-6" />
        
        <div className="bg-gradient-to-b from-[#050810] to-[#100434] rounded-lg border border-[#1a2238] p-6 shadow-lg">
          <h2 className="text-2xl font-semibold mb-6">Edit Profile</h2>
          
          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="grid grid-cols-1 gap-6">
              <div>
                <label htmlFor="name" className="block text-sm font-medium mb-2">
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-md bg-gray-900 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-600"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Email
                </label>
                <div className="flex items-center">
                  <div className="w-full px-4 py-3 rounded-md bg-gray-800/50 border border-gray-700 text-gray-300">
                    {formData.email}
                  </div>
                  <div className="ml-3 text-xs text-gray-400">(Cannot be changed)</div>
                </div>
              </div>
            </div>
            
            <div className="pt-6 border-t border-gray-800">
              <h3 className="text-xl font-medium mb-4">Change Password</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label htmlFor="currentPassword" className="block text-sm font-medium mb-2">
                    Current Password
                  </label>
                  <input
                    type="password"
                    id="currentPassword"
                    name="currentPassword"
                    value={formData.currentPassword}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-md bg-gray-900 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-600"
                  />
                </div>
                
                <div>
                  <label htmlFor="newPassword" className="block text-sm font-medium mb-2">
                    New Password
                  </label>
                  <input
                    type="password"
                    id="newPassword"
                    name="newPassword"
                    value={formData.newPassword}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-md bg-gray-900 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-600"
                  />
                </div>
                
                <div>
                  <label htmlFor="confirmNewPassword" className="block text-sm font-medium mb-2">
                    Confirm New Password
                  </label>
                  <input
                    type="password"
                    id="confirmNewPassword"
                    name="confirmNewPassword"
                    value={formData.confirmNewPassword}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-md bg-gray-900 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-600"
                  />
                </div>
              </div>
            </div>
            
            <div className="flex space-x-4 pt-6">
              <button
                type="submit"
                disabled={isLoading}
                className={`px-8 py-3 rounded-md bg-purple-600 hover:bg-purple-700 transition-colors text-white font-medium ${
                  isLoading ? "opacity-70 cursor-not-allowed" : ""
                }`}
              >
                {isLoading ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </form>
        </div>
        

        <div className="mt-8 bg-gradient-to-b from-[#050810] to-[#100434] rounded-lg border border-[#1a2238] p-6 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4">Account Stats</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-5 bg-gray-900/60 rounded-lg">
              <h3 className="text-lg font-medium text-gray-300 mb-2">Movies in Watchlist</h3>
              <p className="text-3xl font-bold text-purple-400">
                {user?.watchlist?.length || 0}
              </p>
            </div>
            
            <div className="p-5 bg-gray-900/60 rounded-lg">
              <h3 className="text-lg font-medium text-gray-300 mb-2">Movies Watched</h3>
              <p className="text-3xl font-bold text-purple-400">
                {user?.watchedMovies?.length || 0}
              </p>
            </div>
            
            <div className="p-5 bg-gray-900/60 rounded-lg">
              <h3 className="text-lg font-medium text-gray-300 mb-2">Account Created</h3>
              <p className="text-lg">
                {user?.createdAt 
                  ? new Date(user.createdAt).toLocaleDateString() 
                  : 'Unknown'}
              </p>
            </div>
          </div>
        </div>
        
        {/* Danger Zone */}
        <div className="mt-8 mb-12 bg-gradient-to-b from-[#200810] to-[#240424] rounded-lg border border-red-900/40 p-6 shadow-lg">
          <h2 className="text-2xl font-semibold text-red-400 mb-4">Danger Zone</h2>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between border border-red-800/50 rounded-lg p-6">
            <div className="md:max-w-2xl">
              <h3 className="text-xl font-medium mb-2">Delete Account</h3>
              <p className="text-gray-400">
                Once you delete your account, there is no going back. All of your data including your profile,
                watchlist, ratings, and recommendations will be permanently removed.
              </p>
            </div>
            
            <div className="mt-4 md:mt-0">
              <button className="px-6 py-3 bg-red-700 hover:bg-red-800 rounded-md text-white transition-colors">
                Delete Account
              </button>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default AccountPage;