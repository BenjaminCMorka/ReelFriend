import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AccountPage from '../AccountPage';
import { useAuthStore } from '../../store/authStore';
import { toast } from 'react-hot-toast';
import axios from 'axios';


jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn()
}));


jest.mock('axios');


jest.mock('../../components/Navbar', () => () => <div data-testid="navbar-mock">Navbar</div>);


jest.mock('react-hot-toast', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    info: jest.fn()
  }
}));

describe('AccountPage Component', () => {

  const mockUser = {
    _id: 'user123',
    name: 'Test User',
    email: 'test@example.com',
    watchlist: ['123', '456'],
    watchedMovies: [
      { movieId: '789', rating: 4, watchedAt: new Date().toISOString() }
    ],
    createdAt: '2023-01-01T00:00:00.000Z'
  };


  const mockStoreData = {
    user: mockUser,
    isAuthenticated: true,
    checkAuth: jest.fn().mockResolvedValue({})
  };

  beforeEach(() => {

    useAuthStore.mockReturnValue(mockStoreData);
    

    axios.put.mockResolvedValue({ 
      data: { 
        success: true,
        user: mockUser
      } 
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders account page with user data', () => {
    render(<AccountPage />);

    expect(screen.getByText('Account Settings')).toBeInTheDocument();
    

    expect(screen.getByDisplayValue('Test User')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
    

    expect(screen.getByText('Movies in Watchlist')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument(); 
    
    expect(screen.getByText('Movies Watched')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument(); 
    
    expect(screen.getByText('Account Created')).toBeInTheDocument();
    expect(screen.getByText('1/1/2023')).toBeInTheDocument(); 
  });

  test('shows login message for unauthenticated users', () => {

    useAuthStore.mockReturnValue({
      ...mockStoreData,
      isAuthenticated: false,
      user: null
    });
    
    render(<AccountPage />);

    expect(screen.getByText('Please log in to view your account')).toBeInTheDocument();
    

    expect(screen.queryByText('Account Settings')).not.toBeInTheDocument();
  });

  test('handles name update correctly', async () => {
    const user = userEvent.setup();
    render(<AccountPage />);

    const nameInput = screen.getByDisplayValue('Test User');
    await user.clear(nameInput);
    await user.type(nameInput, 'Updated Name');

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await user.click(saveButton);
    
 
    expect(axios.put).toHaveBeenCalledWith(
      'http://localhost:5001/api/auth/update-profile',
      { name: 'Updated Name' },
      { withCredentials: true }
    );
    

    expect(toast.success).toHaveBeenCalledWith('Profile updated successfully!');
    

    expect(mockStoreData.checkAuth).toHaveBeenCalled();
  });

  test('handles password update correctly', async () => {
    const user = userEvent.setup();
    render(<AccountPage />);
    

    const currentPasswordInput = screen.getByLabelText('Current Password');
    const newPasswordInput = screen.getByLabelText('New Password');
    const confirmPasswordInput = screen.getByLabelText('Confirm New Password');
    
    await user.type(currentPasswordInput, 'oldPassword123');
    await user.type(newPasswordInput, 'newPassword123');
    await user.type(confirmPasswordInput, 'newPassword123');
    

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await user.click(saveButton);
    

    expect(axios.put).toHaveBeenCalledWith(
      'http://localhost:5001/api/auth/update-profile',
      { 
        currentPassword: 'oldPassword123',
        newPassword: 'newPassword123'
      },
      { withCredentials: true }
    );
  });

  test('shows error when passwords do not match', async () => {
    const user = userEvent.setup();
    render(<AccountPage />);
    
   
    const currentPasswordInput = screen.getByLabelText('Current Password');
    const newPasswordInput = screen.getByLabelText('New Password');
    const confirmPasswordInput = screen.getByLabelText('Confirm New Password');
    
    await user.type(currentPasswordInput, 'oldPassword123');
    await user.type(newPasswordInput, 'newPassword123');
    await user.type(confirmPasswordInput, 'differentPassword');
    

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await user.click(saveButton);
    

    expect(toast.error).toHaveBeenCalledWith('New password and confirmation do not match');
    

    expect(axios.put).not.toHaveBeenCalled();
  });

  test('shows error when current password is missing for password change', async () => {
    const user = userEvent.setup();
    render(<AccountPage />);
    

    const newPasswordInput = screen.getByLabelText('New Password');
    const confirmPasswordInput = screen.getByLabelText('Confirm New Password');
    
    await user.type(newPasswordInput, 'newPassword123');
    await user.type(confirmPasswordInput, 'newPassword123');
    

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await user.click(saveButton);
    

    expect(toast.error).toHaveBeenCalledWith('Current password is required to set a new password');
    

    expect(axios.put).not.toHaveBeenCalled();
  });

  test('shows info toast when no changes made', async () => {
    const user = userEvent.setup();
    render(<AccountPage />);
    
   
    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await user.click(saveButton);
    

    expect(toast.info).toHaveBeenCalledWith('No changes to save');
    

    expect(axios.put).not.toHaveBeenCalled();
  });

  test('handles API error during update', async () => {

    axios.put.mockRejectedValueOnce({ 
      response: { 
        data: { 
          message: 'Update failed' 
        } 
      } 
    });
    
    const user = userEvent.setup();
    render(<AccountPage />);
    

    const nameInput = screen.getByDisplayValue('Test User');
    await user.clear(nameInput);
    await user.type(nameInput, 'Updated Name');

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await user.click(saveButton);
   
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Update failed');
    });
  });

  test('toggles loading state during API call', async () => {
    const user = userEvent.setup();
    render(<AccountPage />);
    

    const nameInput = screen.getByDisplayValue('Test User');
    await user.clear(nameInput);
    await user.type(nameInput, 'Updated Name');
    

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    

    const clickPromise = user.click(saveButton);
    

    expect(screen.getByRole('button', { name: 'Saving...' })).toBeInTheDocument();
    

    await clickPromise;
    
   
    expect(screen.getByRole('button', { name: 'Save Changes' })).toBeInTheDocument();
  });
});