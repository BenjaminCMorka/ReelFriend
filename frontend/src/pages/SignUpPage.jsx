import { motion } from "framer-motion";
import Input from "../components/Input";
import { Loader, Lock, Mail, User } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { useAuthStore } from "../store/authStore";

const SignUpPage = () => {
	const [name, setName] = useState("");
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");

	const { signup, error, isLoading } = useAuthStore();

	const handleSignUp = async (e) => {
		e.preventDefault();

		try {
			await signup(email, password, name);
		} catch (error) {
			console.log(error);
		}
	};
	return (
		<div className="min-h-screen w-full bg-gray-950 flex flex-col">
			
			<Link
				to="/"
				className="absolute top-6 left-6 text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-700 text-transparent bg-clip-text hover:opacity-80 transition-opacity"
			>
				ReelFriend
			</Link>
		
		<div className="flex justify-center items-center flex-1">
		<motion.div
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.5 }}
			className='max-w-md w-full bg-gray-800 bg-opacity-50 backdrop-filter backdrop-blur-xl rounded-2xl shadow-xl 
			overflow-hidden'
		>
			
			<div className='p-8'>
				
				<h2 className='text-3xl font-bold mb-6 text-center bg-gradient-to-r from-purple-400 to-blue-500 text-transparent bg-clip-text'>
				Lights, Camera, Sign Up!
				</h2>

				<form onSubmit={handleSignUp}>
					<Input
						icon={(props) => <User {...props} className="text-purple-400" />}
						type="text"
						placeholder="What should I call you?"
						value={name}
						onChange={(e) => setName(e.target.value)}
					/>
					<Input
						icon={(props) => <Mail {...props} className="text-purple-400" />}
						type="email"
						placeholder="What's your Email Address?"
						value={email}
						onChange={(e) => setEmail(e.target.value)}
					/>
					<Input
						icon={(props) => <Lock {...props} className="text-purple-400" />}
						type="password"
						placeholder="Choose a secret password - between us!"
						value={password}
						onChange={(e) => setPassword(e.target.value)}
					/>
					{error && <p className='text-red-500 font-semibold mt-2'>{error}</p>}


					<motion.button
						className='mt-5 w-full py-3 px-4 bg-gradient-to-r from-purple-500 to-blue-600 text-white 
						font-bold rounded-lg shadow-lg hover:from-purple-600
						hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2
						 focus:ring-offset-gray-900 transition duration-200'
						whileHover={{ scale: 1.02 }}
						whileTap={{ scale: 0.98 }}
						type='submit'
						disabled={isLoading}
					>
						{isLoading ? <Loader className=' animate-spin mx-auto' size={24} /> : "Let's get started!"}
					</motion.button>
				</form>
			</div>
			<div className='px-8 py-4 bg-gray-900 bg-opacity-50 flex justify-center'>
				<p className='text-sm text-gray-400'>
					Have we met before? Already have an account?{" "}
					<Link to={"/login"} className='text-blue-400 hover:underline'>
						Login
					</Link>
				</p>
			</div>
		</motion.div> </div>
		</div>
	);
};
export default SignUpPage;