import { render, screen } from '@testing-library/react';
import FloatingShape from '../FloatingShape';

describe('FloatingShape Component', () => {
  test('renders floating shape with correct props', () => {
    const props = {
      color: 'bg-purple-600',
      size: 'w-64 h-64',
      top: '10%',
      left: '20%',
      delay: 2
    };
    
    render(<FloatingShape {...props} />);
    
    const shapeElement = screen.getByRole('status', { hidden: true });
    

    expect(shapeElement).toHaveClass('bg-purple-600');
    expect(shapeElement).toHaveClass('w-64');
    expect(shapeElement).toHaveClass('h-64');
    

    expect(shapeElement).toHaveStyle({
      top: '10%',
      left: '20%'
    });
  });

  test('allows different colors and sizes', () => {
    const props = {
      color: 'bg-blue-500',
      size: 'w-48 h-48',
      top: '30%',
      left: '40%',
      delay: 0
    };
    
    render(<FloatingShape {...props} />);
    
    const shapeElement = screen.getByRole('status', { hidden: true });
 
    expect(shapeElement).toHaveClass('bg-blue-500');
    expect(shapeElement).toHaveClass('w-48');
    expect(shapeElement).toHaveClass('h-48');
  });
});