import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Input from '../Input';
import { Mail } from 'lucide-react';


describe('Input Component', () => {
  test('renders input with icon', () => {
    render(
      <Input
        icon={Mail}
        type="email"
        placeholder="Email"
        value="test@example.com"
        onChange={() => {}}
      />
    );
    
    const inputElement = screen.getByPlaceholderText('Email');
    expect(inputElement).toBeInTheDocument();
    expect(inputElement).toHaveValue('test@example.com');
    expect(screen.getByTestId('mail-icon')).toBeInTheDocument();
  });

  test('triggers onChange handler when input changes', async () => {
    const handleChange = jest.fn();
    const user = userEvent.setup();
    
    render(
      <Input
        icon={Mail}
        type="email"
        placeholder="Email"
        value=""
        onChange={handleChange}
      />
    );
    
    const inputElement = screen.getByPlaceholderText('Email');
    await user.type(inputElement, 'test@example.com');
    
    expect(handleChange).toHaveBeenCalled();
  });
});