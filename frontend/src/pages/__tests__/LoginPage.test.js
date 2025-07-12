import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LoginPage from '../LoginPage';
import { useAuthStore } from '../../store/authStore';


jest.mock('react-router-dom', () => ({
  Link: ({ to, children, className }) => (
    <a href={to} className={className} data-testid="mock-link">
      {children}
    </a>
  )
}));


jest.mock('lucide-react', () => ({
  Mail: () => <div data-testid="mail-icon" />,
  Lock: () => <div data-testid="lock-icon" />,
  Loader: () => <div data-testid="loader-icon" />
}));


jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn()
}));


jest.mock('../../components/Input', () => ({ icon, type, placeholder, value, onChange }) => (
  <div className="input-mock">
    {icon && icon({})}
    <input 
      type={type} 
      placeholder={placeholder} 
      value={value} 
      onChange={onChange}
      data-testid={`input-${placeholder}`}
    />
  </div>
));

describe('LoginPage Component', () => {
  const mockLogin = jest.fn();
  
  beforeEach(() => {
    // Setup auth store mock
    useAuthStore.mockReturnValue({
      login: mockLogin,
      isLoading: false,
      error: null
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders login form with correct fields', () => {
    render(<LoginPage />);
    
    // Check for title
    expect(screen.getByText('Welcome Back')).toBeInTheDocument();
    
    // Check for input fields
    expect(screen.getByTestId('input-Email Address')).toBeInTheDocument();
    expect(screen.getByTestId('input-Password')).toBeInTheDocument();
    
    // Check for links
    expect(screen.getByText('Forgot password?')).toBeInTheDocument();
    expect(screen.getByText('Sign up')).toBeInTheDocument();
    
    // Check for login button
    expect(screen.getByRole('button', { name: 'Login' })).toBeInTheDocument();
  });

  test('handles form input changes', async () => {
    const user = userEvent.setup();
    render(<LoginPage />);
    
    const emailInput = screen.getByTestId('input-Email Address');
    const passwordInput = screen.getByTestId('input-Password');
    
    await user.type(emailInput, 'test@example.com');
    await user.type(passwordInput, 'password123');
    
    expect(emailInput).toHaveValue('test@example.com');
    expect(passwordInput).toHaveValue('password123');
  });

  test('calls login function with correct values when form is submitted', async () => {
    const user = userEvent.setup();
    render(<LoginPage />);
    
    const emailInput = screen.getByTestId('input-Email Address');
    const passwordInput = screen.getByTestId('input-Password');
    const loginButton = screen.getByRole('button', { name: 'Login' });
    
    await user.type(emailInput, 'test@example.com');
    await user.type(passwordInput, 'password123');
    await user.click(loginButton);
    
    expect(mockLogin).toHaveBeenCalledWith('test@example.com', 'password123');
  });

  test('displays loading state when isLoading is true', () => {

    useAuthStore.mockReturnValue({
      login: mockLogin,
      isLoading: true,
      error: null
    });
    
    render(<LoginPage />);
    
   
    expect(screen.getByTestId('loader-icon')).toBeInTheDocument();
    expect(screen.queryByText('Login')).not.toBeInTheDocument();
  });

  test('displays error message when login fails', () => {

    useAuthStore.mockReturnValue({
      login: mockLogin,
      isLoading: false,
      error: 'Invalid credentials'
    });
    
    render(<LoginPage />);
    

    expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
  });
});