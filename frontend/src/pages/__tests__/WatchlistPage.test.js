import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import WatchlistPage from '../WatchlistPage';
import { useAuthStore } from '../../store/authStore';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';

jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn(),
  getState: jest.fn()
}));

jest.mock('react-router-dom', () => ({
  useNavigate: jest.fn()
}));

jest.mock('../../components/Navbar', () => {
  const MockNavbar = () => <div data-testid="navbar-mock">Navbar</div>;
  return MockNavbar;
});

jest.mock('react-hot-toast', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn()
  }
}));

const mockGetState = jest.fn();
useAuthStore.getState = mockGetState;

describe('WatchlistPage Component', () => {
  const mockWatchlistMovies = [
    {
      id: 123,
      title: 'Test Movie 1',
      poster_path: '/poster1.jpg',
      release_date: '2023-01-01',
      overview: 'Overview 1',
      genres: [{ name: 'Action' }, { name: 'Drama' }]
    },
    {
      id: 456,
      title: 'Test Movie 2',
      poster_path: '/poster2.jpg',
      release_date: '2023-02-01',
      overview: 'Overview 2',
      genres: [{ name: 'Comedy' }]
    }
  ];

  const mockStoreData = {
    user: { 
      _id: 'user123', 
      name: 'Test User',
      watchlist: ['123', '456']
    },
    isAuthenticated: true,
    removeFromWatchlist: jest.fn().mockResolvedValue({ success: true })
  };

  const mockNavigate = jest.fn();

  const mockRect = {
    top: 100,
    left: 100,
    width: 100,
    height: 100,
    bottom: 200,
    right: 200
  };

  beforeEach(() => {
    useAuthStore.mockReturnValue(mockStoreData);
    useNavigate.mockReturnValue(mockNavigate);
    mockGetState.mockReturnValue({
      markMovieAsWatched: jest.fn().mockResolvedValue({ success: true })
    });

    global.fetch = jest.fn().mockImplementation((url) => {
      if (url.includes('/movie/') && !url.includes('/videos')) {
        const movieId = url.match(/\/movie\/(\d+)/)?.[1];
        const movie = mockWatchlistMovies.find(m => m.id.toString() === movieId);
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(movie || {})
        });
      }

      if (url.includes('/videos')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ 
            results: [
              { type: 'Trailer', key: 'abc123' }
            ] 
          })
        });
      }

      return Promise.resolve({
        ok: false,
        json: () => Promise.resolve({})
      });
    });

    Element.prototype.getBoundingClientRect = jest.fn().mockReturnValue(mockRect);

    Object.defineProperty(window, 'scrollY', {
      configurable: true,
      value: 0
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders watchlist with trailer links when available', async () => {
    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      const trailerButtons = screen.getAllByText('Watch Trailer');
      expect(trailerButtons.length).toBeGreaterThan(0);
    });

    const trailerLinks = screen.getAllByRole('link');
    expect(trailerLinks[0]).toHaveAttribute('href', expect.stringContaining('youtube.com'));
  });

  test('renders empty watchlist message when watchlist is empty', async () => {
    useAuthStore.mockReturnValue({
      ...mockStoreData,
      user: { ...mockStoreData.user, watchlist: [] }
    });

    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Your watchlist is empty')).toBeInTheDocument();
    });

    const discoverButton = screen.getByRole('button', { name: 'Discover Movies' });
    fireEvent.click(discoverButton);
    expect(mockNavigate).toHaveBeenCalledWith('/');
  });

  test('renders empty state when user is not authenticated', async () => {
    useAuthStore.mockReturnValue({
      ...mockStoreData,
      isAuthenticated: false
    });

    await act(async () => {
      render(<WatchlistPage />);
    });

    expect(screen.queryByText(/Loading your watchlist/i)).not.toBeInTheDocument();
    expect(screen.getByText('Your watchlist is empty')).toBeInTheDocument();
  });

  test('handles remove from watchlist functionality', async () => {
    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    const removeButtons = screen.getAllByTitle('Remove from watchlist');
    fireEvent.click(removeButtons[0]);

    expect(mockStoreData.removeFromWatchlist).toHaveBeenCalled();

    await waitFor(() => {
      expect(toast.success).toHaveBeenCalledWith(expect.stringContaining('Removed'));
    });
  });

  test('handles error when removing from watchlist', async () => {
    useAuthStore.mockReturnValue({
      ...mockStoreData,
      removeFromWatchlist: jest.fn().mockResolvedValue({ success: false, message: 'Remove failed' })
    });

    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    const removeButtons = screen.getAllByTitle('Remove from watchlist');
    fireEvent.click(removeButtons[0]);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Remove failed');
    });
  });

  test('redirects to login when removing from watchlist while not authenticated', async () => {
    const mockRemoveFromWatchlist = jest.fn();
    useAuthStore.mockReturnValue({
      ...mockStoreData,
      isAuthenticated: true,
      removeFromWatchlist: mockRemoveFromWatchlist
    });

    const { rerender } = render(<WatchlistPage />);

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    useAuthStore.mockReturnValue({
      ...mockStoreData,
      isAuthenticated: false,
      removeFromWatchlist: mockRemoveFromWatchlist
    });

    rerender(<WatchlistPage />);

    mockNavigate.mockClear();
    toast.error.mockClear();

    const removeButtons = screen.getAllByTitle('Remove from watchlist');
    await act(async () => {
      fireEvent.click(removeButtons[0]);
    });

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/login');
      expect(toast.error).toHaveBeenCalledWith('Please login to manage your watchlist');
    });
  });

  test('toggles movie description when button clicked', async () => {
    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    expect(screen.queryByText('Overview 1')).not.toBeInTheDocument();

    const descButtons = screen.getAllByTitle('Show Description');
    fireEvent.click(descButtons[0]);

    expect(screen.getByText('Overview 1')).toBeInTheDocument();

    fireEvent.click(descButtons[0]);

    await waitFor(() => {
      expect(screen.queryByText('Overview 1')).not.toBeInTheDocument();
    });
  });

  test('opens rating modal and submits rating', async () => {
    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    const watchedButtons = screen.getAllByTitle("I've Watched This");
    fireEvent.click(watchedButtons[0]);

    expect(screen.getByText('Rate this movie')).toBeInTheDocument();

    const stars = screen.getAllByText('★');
    fireEvent.click(stars[3]); 

    const submitButton = screen.getByText('Submit');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(useAuthStore.getState().markMovieAsWatched).toHaveBeenCalled();
    });

    expect(toast.success).toHaveBeenCalled();
  });

  test('handles cancel button in rating modal', async () => {
    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    const watchedButtons = screen.getAllByTitle("I've Watched This");
    fireEvent.click(watchedButtons[0]);

    expect(screen.getByText('Rate this movie')).toBeInTheDocument();

    const cancelButton = screen.getByText('Cancel');
    fireEvent.click(cancelButton);

    await waitFor(() => {
      expect(screen.queryByText('Rate this movie')).not.toBeInTheDocument();
    });
  });

  test('handles rating submission error', async () => {
    mockGetState.mockReturnValue({
      markMovieAsWatched: jest.fn().mockRejectedValue(new Error('Rating failed'))
    });

    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    const watchedButtons = screen.getAllByTitle("I've Watched This");
    fireEvent.click(watchedButtons[0]);

    expect(screen.getByText('Rate this movie')).toBeInTheDocument();

    const stars = screen.getAllByText('★');
    fireEvent.click(stars[2]); 

    const submitButton = screen.getByText('Submit');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Failed to submit rating. Please try again.');
    });
  });

  global.fetch = jest.fn((url) => {
    if (url.includes('/videos')) {
      return Promise.resolve({
        json: () => Promise.resolve({
          results: [{ type: 'Trailer', key: 'abc123' }],
        }),
      });
    }

    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        id: 1,
        title: 'Test Movie',
        release_date: '2020-01-01',
        genres: [{ name: 'Action' }],
        poster_path: '/poster.jpg',
        overview: 'A test movie',
      }),
    });
  });

  test('opens rating modal and sets modal position', async () => {
    const { getAllByTitle } = render(<WatchlistPage />);

    await waitFor(() => expect(getAllByTitle("I've Watched This")[0]).toBeInTheDocument());

    const mockButton = document.createElement('button');
    mockButton.getBoundingClientRect = () => ({
      top: 200,
      left: 100, 
      width: 0,
      height: 0,
      bottom: 0,
      right: 0,
    });

    fireEvent.click(getAllByTitle("I've Watched This")[0], { currentTarget: mockButton });

    await waitFor(() => {
      expect(document.querySelector('h2')).toHaveTextContent('Rate this movie');
    });
  });

  test('handles error when fetching watchlist movies', async () => {
    global.fetch = jest.fn().mockImplementation(() => {
      return Promise.reject(new Error('Failed to fetch'));
    });

    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Failed to load your watchlist. Please try again later.')).toBeInTheDocument();
    });
  });

  test('logs error when trailer fetch fails', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    global.fetch = jest.fn((url) => {
      if (url.includes('/videos')) {
        return Promise.reject(new Error('Trailer fetch failed'));
      }

      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          id: 1,
          title: 'Movie',
          release_date: '2020-01-01',
          genres: [{ name: 'Drama' }],
          poster_path: '/poster.jpg',
          overview: 'Overview',
        }),
      });
    });

    render(<WatchlistPage />);

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        'Error fetching trailer:',
        expect.any(Error)
      );
    });

    consoleSpy.mockRestore();
  });

  test('handles exception when removing from watchlist', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    useAuthStore.mockReturnValue({
      ...mockStoreData,
      removeFromWatchlist: jest.fn().mockRejectedValue(new Error('Network error'))
    });

    await act(async () => {
      render(<WatchlistPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
    });

    const removeButtons = screen.getAllByTitle('Remove from watchlist');
    await act(async () => {
      fireEvent.click(removeButtons[0]);
    });

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Error removing from watchlist:', 
      expect.any(Error)
    );

    expect(toast.error).toHaveBeenCalledWith('Failed to remove from watchlist. Please try again.');

    consoleErrorSpy.mockRestore();
  });

});
