import { useState } from "react";
import { motion } from "framer-motion";
import { Mail, Lock, Loader } from "lucide-react";
import { Link } from "react-router-dom";
import Input from "../components/Input";
import { useAuthStore } from "../store/authStore";

const LoginPage = () => {
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");

	const { login, isLoading, error } = useAuthStore();

	const handleLogin = async (e) => {
		e.preventDefault();
		await login(email, password);
	};

	return (
		<div className="min-h-screen w-full bg-gray-950 flex flex-col">
			{/* ReelFriend Text (Now Positioned at the Top Left of the Whole Page) */}
			<Link
				to="/"
				className="absolute top-6 left-6 text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-700 text-transparent bg-clip-text hover:opacity-80 transition-opacity"
			>
				ReelFriend
			</Link>

			{/* Centering the Login Box */}
			<div className="flex justify-center items-center flex-1">
				<motion.div
					initial={{ opacity: 0, y: 20 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ duration: 0.5 }}
					className="max-w-md w-full bg-gray-900 bg-opacity-50 backdrop-filter backdrop-blur-xl rounded-2xl shadow-xl overflow-hidden"
				>
					<div className="p-8">
						<h2 className="text-3xl font-bold mb-6 text-center bg-gradient-to-r from-blue-400 to-purple-700 text-transparent bg-clip-text">
							Welcome Back
						</h2>

						<form onSubmit={handleLogin}>
							<Input
								icon={(props) => <Mail {...props} className="text-purple-400" />}
								type="email"
								placeholder="Email Address"
								value={email}
								onChange={(e) => setEmail(e.target.value)}
							/>

							<Input
								icon={(props) => <Lock {...props} className="text-purple-400" />}
								type="password"
								placeholder="Password"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
							/>

							<div className="flex items-center mb-6">
								<Link to="/forgot-password" className="text-sm text-blue-400 hover:underline">
									Forgot password?
								</Link>
							</div>
							{error && <p className="text-red-500 font-semibold mb-2">{error}</p>}

							<motion.button
								whileHover={{ scale: 1.02 }}
								whileTap={{ scale: 0.98 }}
								className="w-full py-3 px-4 bg-gradient-to-r from-purple-500 to-blue-600 text-white font-bold rounded-lg shadow-lg hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition duration-200"
								type="submit"
								disabled={isLoading}
							>
								{isLoading ? <Loader className="w-6 h-6 animate-spin mx-auto" /> : "Login"}
							</motion.button>
						</form>
					</div>
					<div className="px-8 py-4 bg-gray-900 bg-opacity-50 flex justify-center">
						<p className="text-sm text-gray-400">
							We haven't met before? Don't have an account?{" "}
							<Link to="/signup" className="text-blue-400 hover:underline">
								Sign up
							</Link>
						</p>
					</div>
				</motion.div>
			</div>
		</div>
	);
};

export default LoginPage;
