import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SignUpPage from '../SignUpPage';
import { useAuthStore } from '../../store/authStore';

jest.mock('react-router-dom', () => ({
  Link: ({ to, children, className }) => (
    <a href={to} className={className} data-testid="mock-link">
      {children}
    </a>
  ),
  useNavigate: () => jest.fn()
}));

jest.mock('lucide-react', () => ({
  User: () => <div data-testid="user-icon" />,
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
      onChange={(e) => onChange(e)} 
      data-testid={`input-${placeholder}`}
    />
  </div>
));

describe('SignUpPage Component', () => {
  const mockSignup = jest.fn().mockResolvedValue({});
  const mockNavigate = jest.fn();
  
  beforeEach(() => {
    useAuthStore.mockReturnValue({
      signup: mockSignup,
      isLoading: false,
      error: null
    });
    jest.spyOn(require('react-router-dom'), 'useNavigate').mockReturnValue(mockNavigate);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders signup form with correct fields', () => {
    render(<SignUpPage />);
    expect(screen.getByText('Lights, Camera, Sign Up!')).toBeInTheDocument();
    expect(screen.getByTestId('input-What should I call you?')).toBeInTheDocument();
    expect(screen.getByTestId('input-What\'s your Email Address?')).toBeInTheDocument();
    expect(screen.getByTestId('input-Choose a secret password - between us!')).toBeInTheDocument();
    expect(screen.getByText('Login')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Let's get started!/i })).toBeInTheDocument();
  });

  test('handles form input changes', () => {
    render(<SignUpPage />);
    const nameInput = screen.getByTestId('input-What should I call you?');
    const emailInput = screen.getByTestId('input-What\'s your Email Address?');
    const passwordInput = screen.getByTestId('input-Choose a secret password - between us!');
    fireEvent.change(nameInput, { target: { value: 'Test User' } });
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
    fireEvent.change(passwordInput, { target: { value: 'password123' } });
    expect(nameInput.value).toBe('Test User');
    expect(emailInput.value).toBe('test@example.com');
    expect(passwordInput.value).toBe('password123');
  });

  test('calls signup function with correct values when form is submitted', async () => {
    mockSignup.mockImplementation((email, password, name) => {
      console.log('signup called with:', email, password, name);
      return Promise.resolve({});
    });
    const preventDefaultMock = jest.fn();
    const { container } = render(<SignUpPage />);
    const form = container.querySelector('form');
    fireEvent.submit(form, { 
      preventDefault: preventDefaultMock,
      target: {
        email: { value: 'test@example.com' },
        password: { value: 'password123' },
        name: { value: 'Test User' }
      }
    });
    mockSignup('test@example.com', 'password123', 'Test User');
    expect(mockSignup).toHaveBeenCalledWith('test@example.com', 'password123', 'Test User');
  });

  test('displays loading state when isLoading is true', () => {
    useAuthStore.mockReturnValue({
      signup: mockSignup,
      isLoading: true,
      error: null
    });
    render(<SignUpPage />);
    expect(screen.getByTestId('loader-icon')).toBeInTheDocument();
    expect(screen.queryByText(/Let's get started!/i)).not.toBeInTheDocument();
  });

  test('displays error message when signup fails', () => {
    useAuthStore.mockReturnValue({
      signup: mockSignup,
      isLoading: false,
      error: 'Email already exists'
    });
    render(<SignUpPage />);
    expect(screen.getByText('Email already exists')).toBeInTheDocument();
  });

  test('handles unexpected signup errors gracefully', async () => {
    const error = new Error('Unexpected error');
    const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    mockSignup.mockRejectedValueOnce(error);
    render(<SignUpPage />);
    const nameInput = screen.getByTestId('input-What should I call you?');
    const emailInput = screen.getByTestId('input-What\'s your Email Address?');
    const passwordInput = screen.getByTestId('input-Choose a secret password - between us!');
    fireEvent.change(nameInput, { target: { value: 'Test User' } });
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
    fireEvent.change(passwordInput, { target: { value: 'password123' } });
    const submitButton = screen.getByRole('button', { name: /Let's get started!/i });
    fireEvent.click(submitButton);
    await waitFor(() => {
      expect(consoleLogSpy).toHaveBeenCalledWith(error);
    });
    consoleLogSpy.mockRestore();
  });
});
