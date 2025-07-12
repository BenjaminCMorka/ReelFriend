import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ForgotPasswordPage from '../ForgotPasswordPage';
import { BrowserRouter } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import toast from 'react-hot-toast';


jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    error: jest.fn(),
    success: jest.fn(),
  },
}));

jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn(),
}));

const mockResetPassword = jest.fn();
const mockNavigate = jest.fn();


jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

describe('ForgotPasswordPage', () => {
  beforeEach(() => {
    useAuthStore.mockReturnValue({
      isLoading: false,
      resetPassword: mockResetPassword,
    });
  });

  const setup = () => {
    render(
      <BrowserRouter>
        <ForgotPasswordPage />
      </BrowserRouter>
    );
  };

  test('renders all input fields and button', () => {
    setup();
    expect(screen.getByPlaceholderText('Email Address')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('New Password')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Confirm New Password')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset password/i })).toBeInTheDocument();
  });

  test('displays error when passwords do not match', async () => {
    setup();
    fireEvent.change(screen.getByPlaceholderText('Email Address'), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('New Password'), { target: { value: 'abc123' } });
    fireEvent.change(screen.getByPlaceholderText('Confirm New Password'), { target: { value: 'xyz456' } });

    fireEvent.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Passwords do not match');
    });
  });

  test('calls resetPassword and shows success toast on match', async () => {
    mockResetPassword.mockResolvedValueOnce();
    setup();
    fireEvent.change(screen.getByPlaceholderText('Email Address'), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('New Password'), { target: { value: 'password123' } });
    fireEvent.change(screen.getByPlaceholderText('Confirm New Password'), { target: { value: 'password123' } });

    fireEvent.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(mockResetPassword).toHaveBeenCalledWith('test@example.com', 'password123');
      expect(toast.success).toHaveBeenCalledWith('Password reset successfully!');
    });
  });

  test('shows error toast on reset failure', async () => {
    mockResetPassword.mockRejectedValueOnce(new Error('Network error'));
    setup();
    fireEvent.change(screen.getByPlaceholderText('Email Address'), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('New Password'), { target: { value: 'password123' } });
    fireEvent.change(screen.getByPlaceholderText('Confirm New Password'), { target: { value: 'password123' } });

    fireEvent.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Network error');
    });
  });


  test('disables submit button when loading', () => {
    useAuthStore.mockReturnValue({
      isLoading: true,
      resetPassword: jest.fn(),
    });
    setup();
    expect(screen.getByRole('button', { name: /resetting.../i })).toBeDisabled();
  });
});
