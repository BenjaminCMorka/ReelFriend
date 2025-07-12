import { Navigate, Route, Routes } from "react-router-dom";
import FloatingShape from "./components/FloatingShape";

import SignUpPage from "./pages/SignUpPage";
import LoginPage from "./pages/LoginPage";
import DashboardPage from "./pages/DashboardPage";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";
import LandingPage from "./pages/LandingPage";
import WatchlistPage from "./pages/WatchlistPage";
import AboutPage from "./pages/AboutPage";
import AccountPage from "./pages/AccountPage";
import SearchResultsPage from "./pages/SearchResultsPage";

import LoadingSpinner from "./components/LoadingSpinner";

import { Toaster } from "react-hot-toast";
import { useAuthStore } from "./store/authStore";
import { useEffect } from "react";
import OnboardingPage from "./pages/OnboardingPage";

// protect routes which need authentication
const ProtectedRoute = ({ children }) => {
	const { isAuthenticated } = useAuthStore();

	if (!isAuthenticated) {
		return <Navigate to='/login' replace />;
	}


	return children;
};


const ProtectLanding = ({ children }) => {
	const { isAuthenticated } = useAuthStore();

	if (isAuthenticated) {
		return <Navigate to="/dashboard" replace />;
	}

	return children;
};
const ProtectDashboard = ({ children }) => {
	const { isAuthenticated, user } = useAuthStore();

	if (!isAuthenticated || !user.hasOnboarded) {
		return <Navigate to="/login" replace />;
	}

	return children;
};

const ProtectAuth = ({ children }) => {
	const { isAuthenticated } = useAuthStore();

	if (isAuthenticated) {
		return <Navigate to="/dashboard" replace />;
	}

	return children;
};


// prevent authenticated users from accessing auth pages
const RedirectAuthenticatedUser = ({ children }) => {
	const { isAuthenticated, user } = useAuthStore();

	if (isAuthenticated && !user.hasOnboarded) {
		return <Navigate to='/onboard' replace />;
	}
	else if (isAuthenticated && user.hasOnboarded){
		return <Navigate to='/dashboard' replace />;
	}

	return children;
};




function App() {
	const { isCheckingAuth, checkAuth } = useAuthStore();




	useEffect(() => {
		checkAuth();
	}, [checkAuth]);

	if (isCheckingAuth) return <LoadingSpinner />;

	return (
		<div className='min-h-screen bg-gradient-to-br from-gray-950 via-gray-950-900 to-gray-950 flex items-center justify-center relative overflow-hidden'>
			<FloatingShape color='bg-purple-600' size='w-64 h-64' top='-5%' left='10%' delay={0} />
			<FloatingShape color='bg-blue-500' size='w-64 h-64' top='70%' left='80%' delay={5} />
			<FloatingShape color='bg-purple-600' size='w-48 h-48' top='40%' left='-10%' delay={2} />

			<Routes>

				<Route path='/' element={<ProtectLanding><LandingPage /></ProtectLanding>} />

				{/* only when not logged in */}
				<Route path='/signup' element={<RedirectAuthenticatedUser><SignUpPage /></RedirectAuthenticatedUser>} />
				<Route path='/login' element={<RedirectAuthenticatedUser><ProtectAuth><LoginPage /> </ProtectAuth></RedirectAuthenticatedUser>} />
				<Route path='/reset-password' element={<RedirectAuthenticatedUser><ForgotPasswordPage /></RedirectAuthenticatedUser>} />

				{/* only when logged in */}
				<Route path='/onboard' element={<ProtectedRoute><OnboardingPage /></ProtectedRoute>} />
				<Route path='/dashboard' element={<ProtectDashboard><DashboardPage/> </ProtectDashboard>} />
				<Route path='/watchlist' element={<ProtectedRoute><WatchlistPage/></ProtectedRoute>} />
				<Route path='/about' element={<ProtectedRoute><AboutPage/></ProtectedRoute>} />
				<Route path='/account' element={<ProtectedRoute><AccountPage/></ProtectedRoute>} />
				<Route path="/search/:query" element={<SearchResultsPage />} />

				{/* if its an unknown path go to landing page */}
				<Route path='*' element={<Navigate to='/' replace />} />
			</Routes>

			<Toaster />
		</div>
	);
}

export default App;
