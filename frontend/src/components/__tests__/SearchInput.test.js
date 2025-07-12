import { render, screen, fireEvent } from '@testing-library/react';
import SearchInput from '../SearchInput';

describe('SearchInput component', () => {
  test('renders without crashing', () => {
    render(<SearchInput />);
    const input = screen.getByRole('textbox');
    expect(input).toBeInTheDocument();
  });

  test('applies custom class names', () => {
    render(<SearchInput className="custom-class" />);
    const input = screen.getByRole('textbox');
    expect(input).toHaveClass('custom-class');
  });

  test('passes through additional props', () => {
    render(<SearchInput placeholder="Search..." value="test" readOnly />);
    const input = screen.getByPlaceholderText('Search...');
    expect(input.value).toBe('test');
    expect(input).toHaveAttribute('readonly');
  });

  test('calls onChange when typing', () => {
    const handleChange = jest.fn();
    render(<SearchInput onChange={handleChange} />);
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'hello' } });
    expect(handleChange).toHaveBeenCalledTimes(1);
  });
});
