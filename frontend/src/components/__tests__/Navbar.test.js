import { render, screen, fireEvent } from '@testing-library/react';
import Navbar from '../Navbar';
import { useAuthStore } from '../../store/authStore';
import { useLocation, useNavigate } from 'react-router-dom';


jest.mock('react-router-dom', () => ({
  Link: ({ to, children, className }) => (
    <a href={to} className={className} data-testid="mock-link">
      {children}
    </a>
  ),
  useLocation: jest.fn(),
  useNavigate: jest.fn(),
}));


jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn(),
}));


jest.mock('react-icons/fa', () => ({
  FaUserCircle: () => <div data-testid="user-icon" />,
}));

describe('Navbar Component', () => {
  const mockLogout = jest.fn();
  const mockNavigate = jest.fn();
  
  beforeEach(() => {
    useAuthStore.mockReturnValue({
      logout: mockLogout,
    });
    useLocation.mockReturnValue({ pathname: '/dashboard' });
    useNavigate.mockReturnValue(mockNavigate);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders navbar with correct links', () => {
    render(<Navbar />);
    
    expect(screen.getByText('ReelFriend')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('My Watchlist')).toBeInTheDocument();
    expect(screen.getByText('About')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search movies...')).toBeInTheDocument();
  });

  test('handles search functionality', () => {
    render(<Navbar />);
    
    const searchInput = screen.getByPlaceholderText('Search movies...');
    fireEvent.change(searchInput, { target: { value: 'Avengers' } });
    fireEvent.keyDown(searchInput, { key: 'Enter' });
    
    expect(mockNavigate).toHaveBeenCalledWith('/search/Avengers');
  });

  test('handles logout', () => {
    render(<Navbar />);
    
    const userIcon = screen.getByTestId('user-icon');
    fireEvent.mouseEnter(userIcon);
    
    const logoutButton = screen.getByText('Logout');
    fireEvent.click(logoutButton);
    
    expect(mockLogout).toHaveBeenCalled();
  });

  test('changes background on scroll', () => {
    render(<Navbar />);
    

    const navbar = screen.getByRole('navigation');
    expect(navbar).toHaveClass('bg-transparent');
    

    window.scrollY = 100;
    fireEvent.scroll(window);
    
 
    expect(navbar).toHaveClass('bg-black');
  });
});