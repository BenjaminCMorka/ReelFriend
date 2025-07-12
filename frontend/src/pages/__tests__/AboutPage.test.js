import { render, screen } from '@testing-library/react';
import AboutPage from '../AboutPage';

jest.mock('../../components/Navbar', () => () => <div data-testid="navbar-mock">Navbar</div>);

describe('AboutPage Component', () => {
  test('renders about page with correct content', () => {
    render(<AboutPage />);
    

    expect(screen.getByTestId('navbar-mock')).toBeInTheDocument();
    

    expect(screen.getByText('About ReelFriend')).toBeInTheDocument();
    
 
    expect(screen.getByText(/ReelFriend is here to help you to discover movies you will love/i)).toBeInTheDocument();
    

    expect(screen.getByText('Explainability & Transparency')).toBeInTheDocument();
    expect(screen.getByText(/ReelFriend isn't a black box/i)).toBeInTheDocument();
  });
});