import Navbar from "../components/Navbar";
import { useState } from "react";

const AccountPage = () => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    currentPassword: "",
    newPassword: "",
    confirmNewPassword: "",
  });

  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    setSuccessMessage("");

    // Basic validation
    if (!formData.currentPassword || !formData.newPassword || !formData.confirmNewPassword) {
      setError("All fields are required.");
      return;
    }

    if (formData.newPassword !== formData.confirmNewPassword) {
      setError("New password and confirmation do not match.");
      return;
    }

    // Placeholder for API call to update user data
    setSuccessMessage("Profile updated successfully!");
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center text-white">
      <Navbar />
      
      <div className="w-full max-w-lg p-6 space-y-6  rounded-lg shadow-md">
        {/* Title */}
        <h1 className="text-3xl font-bold text-center">Edit Profile</h1>

        {/* Error/Success Message */}
        {error && <p className="text-red-500 text-center">{error}</p>}
        {successMessage && <p className="text-green-500 text-center">{successMessage}</p>}

        {/* Profile Edit Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col">
            <label htmlFor="username" className="text-sm">Username</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="mt-2 px-4 py-2 rounded-md bg-gray-700 text-white"
              placeholder="Enter new username"
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="email" className="text-sm">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="mt-2 px-4 py-2 rounded-md bg-gray-700 text-white"
              placeholder="Enter new email"
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="currentPassword" className="text-sm">Current Password</label>
            <input
              type="password"
              id="currentPassword"
              name="currentPassword"
              value={formData.currentPassword}
              onChange={handleChange}
              className="mt-2 px-4 py-2 rounded-md bg-gray-700 text-white"
              placeholder="Enter current password"
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="newPassword" className="text-sm">New Password</label>
            <input
              type="password"
              id="newPassword"
              name="newPassword"
              value={formData.newPassword}
              onChange={handleChange}
              className="mt-2 px-4 py-2 rounded-md bg-gray-700 text-white"
              placeholder="Enter new password"
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="confirmNewPassword" className="text-sm">Confirm New Password</label>
            <input
              type="password"
              id="confirmNewPassword"
              name="confirmNewPassword"
              value={formData.confirmNewPassword}
              onChange={handleChange}
              className="mt-2 px-4 py-2 rounded-md bg-gray-700 text-white"
              placeholder="Confirm new password"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full bg-blue-600 py-2 rounded-md hover:bg-blue-700 transition"
          >
            Save Changes
          </button>
        </form>

        {/* Cancel Button (Optional) */}
        <button
          onClick={() => {}}
          className="w-full text-gray-400 mt-4 py-2 rounded-md hover:bg-gray-600 transition"
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default AccountPage;
