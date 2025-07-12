import { render, screen } from '@testing-library/react';
import LandingPage from '../LandingPage';

// Mock react-router-dom
jest.mock('react-router-dom', () => ({
  Link: ({ to, children, className }) => (
    <a href={to} className={className} data-testid="mock-link">
      {children}
    </a>
  )
}));

describe('LandingPage Component', () => {
  test('renders landing page with correct content', () => {
    render(<LandingPage />);
    
    // Check for main heading and subtitle
    expect(screen.getByText('Welcome to ReelFriend')).toBeInTheDocument();
    expect(screen.getByText(/Looking for something new to watch/i)).toBeInTheDocument();
    
    // Check for buttons
    const getStartedLink = screen.getByText('Get Started');
    const loginLink = screen.getByText('Log In');
    
    expect(getStartedLink).toBeInTheDocument();
    expect(loginLink).toBeInTheDocument();
    
    // Check that links point to correct routes
    expect(getStartedLink.closest('a')).toHaveAttribute('href', '/signup');
    expect(loginLink.closest('a')).toHaveAttribute('href', '/login');
  });
});