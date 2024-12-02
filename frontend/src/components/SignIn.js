import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './styles/SignUp.css'; 


const SignIn = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Assume successful login
      navigate('/dashboard');
    } catch (err) {
      setError('Invalid email or password');
    }
  };

  return (
    <div className="signin-page">
      <div className="signin-container">
        <div className="logo-container">
          <Link to="/" className="logo-title">FlickFest</Link>
        </div>
        <h1>Sign In</h1>

        <form className="signin-form" onSubmit={handleSubmit}>
          <input
            type="email"
            placeholder="Email Address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit" className="signin-button">Sign In</button>
          {error && <p className="error-message">{error}</p>}
        </form>

        <p className="signin-link">
          Don’t have an account? <Link to="/signup">Sign up here</Link>.
        </p>
      </div>
    </div>
  );
};

export default SignIn;
