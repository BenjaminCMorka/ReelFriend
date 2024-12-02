import React, { useState, useEffect  } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import './styles/SignUp.css'; 


const SignUp = () => {

  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [isSignedUp, setIsSignedUp] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (isSignedUp) {
      navigate('/dashboard');
    }
  }, [isSignedUp, navigate]);


  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === 'username') setUsername(value);
    if (name === 'email') setEmail(value);
    if (name === 'password') setPassword(value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
  

    const userData = { username, email, password };
  
    try {

      const response = await axios.post('http://localhost:5001/api/auth/signup', userData);
      
      console.log(response.data); 
  

      setMessage(response.data.message);
      setUsername('');
      setEmail('');
      setPassword('');
      setIsSignedUp(true);
    } catch (err) {
      
      setError(err.response ? err.response.data.error : 'Server error');
    }
  };
  

  return (
    <div className="signup-page">
      <div className="signup-container">
        <div className="logo-container">
        <Link to="/" className="logo-title">
            <h1>flickFest</h1>
          </Link>
        </div>
        <h1>Sign Up</h1>

        {message && <div className="success-message">{message}</div>}
        {error && <div className="error-message">{error}</div>}


        <form className="signup-form" onSubmit={handleSubmit}>
          <input
            type="text"
            name="username"
            value={username}
            placeholder="Username"
            onChange={handleChange}
            required
          />
          <input
            type="email"
            name="email"
            value={email}
            placeholder="Email Address"
            onChange={handleChange}
            required
          />
          <input
            type="password"
            name="password"
            value={password}
            placeholder="Password"
            onChange={handleChange}
            required
          />
          <button type="submit" className="signup-submit-button">Create Account</button>
        </form>
      </div>
    </div>
  );
};

export default SignUp;
