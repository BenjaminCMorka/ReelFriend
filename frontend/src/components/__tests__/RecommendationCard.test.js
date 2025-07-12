import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RecommendationCard from '../RecommendationCard';
import { toast } from 'react-hot-toast';

// Mocks
jest.mock('react-hot-toast', () => ({
  toast: {
    error: jest.fn(),
    success: jest.fn(),
  },
}));

const defaultMovie = {
  id: 1,
  title: 'Inception',
  release_date: '2010-07-16',
  poster_path: '/inception.jpg',
  genres: ['Action', 'Sci-Fi'],
  overview: 'A mind-bending thriller.'
};

const renderComponent = (props = {}) => {
  const defaultProps = {
    movie: defaultMovie,
    explanation: 'Because you liked sci-fi movies.',
    trailer: 'https://youtube.com/trailer',
    isInWatchlist: jest.fn(() => false),
    addToWatchlist: jest.fn(),
    openRatingModal: jest.fn(),
    ...props
  };

  return render(<RecommendationCard {...defaultProps} />);
};

describe('RecommendationCard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('does not render if no movie is provided', () => {
    const { container } = render(<RecommendationCard />);
    expect(container.firstChild).toBeNull();
  });

  test('renders movie title, year, genres', () => {
    renderComponent();
    expect(screen.getByText('Inception')).toBeInTheDocument();
    expect(screen.getByText('2010')).toBeInTheDocument();
    expect(screen.getByText('Action, Sci-Fi')).toBeInTheDocument();
  });

  test('shows trailer button if trailer is provided', () => {
    renderComponent();
    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://youtube.com/trailer');
    expect(screen.getByText(/watch trailer/i)).toBeInTheDocument();
  });

  test('shows fallback if trailer is not provided', () => {
    renderComponent({ trailer: null });
    expect(screen.getByText(/no trailer available/i)).toBeInTheDocument();
  });

  test('handles missing movie ID', async () => {
    const movie = { ...defaultMovie };
    delete movie.id;
    renderComponent({ movie });

    fireEvent.click(screen.getByTitle('Add to Watchlist'));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Movie information is not available');
    });
  });

  test('handles already in watchlist', async () => {
    renderComponent({
      isInWatchlist: () => true
    });

    fireEvent.click(screen.getByTitle('Add to Watchlist'));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('"Inception" is already in your watchlist!');
    });
  });

  test('handles add to watchlist success', async () => {
    const addToWatchlist = jest.fn(() => Promise.resolve());
    renderComponent({ addToWatchlist });

    fireEvent.click(screen.getByTitle('Add to Watchlist'));

    await waitFor(() => {
      expect(addToWatchlist).toHaveBeenCalledWith('1', 'Inception', '/inception.jpg');
      expect(toast.success).toHaveBeenCalledWith('Added "Inception" to your watchlist!');
    });
  });

  test('handles add to watchlist failure', async () => {
    const addToWatchlist = jest.fn(() => Promise.reject(new Error('Network error')));
    renderComponent({ addToWatchlist });

    fireEvent.click(screen.getByTitle('Add to Watchlist'));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Failed to add to watchlist. Please try again.');
    });
  });

  test('calls openRatingModal on rating button click', () => {
    const openRatingModal = jest.fn();
    renderComponent({ openRatingModal });

    fireEvent.click(screen.getByTitle("I've Watched This"));

    expect(openRatingModal).toHaveBeenCalledWith(1, expect.any(Object));
  });

  test('toggles movie description', () => {
    renderComponent();
    const button = screen.getByTitle('Show Description');

    fireEvent.click(button);
    expect(screen.getByText(defaultMovie.overview)).toBeInTheDocument();

    fireEvent.click(button);
    expect(screen.queryByText(defaultMovie.overview)).not.toBeInTheDocument();
  });

  test('toggles explanation section', () => {
    renderComponent();
    const button = screen.getByTitle('Why Recommended');

    fireEvent.click(button);
    expect(screen.getByText(/because you liked sci-fi/i)).toBeInTheDocument();

    fireEvent.click(button);
    expect(screen.queryByText(/because you liked sci-fi/i)).not.toBeInTheDocument();
  });
});
