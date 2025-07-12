import { render, screen } from '@testing-library/react';
import LoadingSpinner from '../LoadingSpinner';


describe('LoadingSpinner Component', () => {
  test('renders loading spinner', () => {
    render(<LoadingSpinner />);
    
    const spinnerElement = screen.getByRole('status', { hidden: true });
    expect(spinnerElement).toBeInTheDocument();
    expect(spinnerElement).toHaveClass('border-t-green-500');
  });
});