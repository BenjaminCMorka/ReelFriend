import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DashboardPage from '../DashboardPage';
import { useAuthStore } from '../../store/authStore';
import axios from 'axios';

jest.mock('../../store/authStore', () => {
  const mockMarkMovieAsWatched = jest.fn().mockResolvedValue({ success: true });
  const mockStore = {
    useAuthStore: jest.fn(),
    getState: jest.fn().mockReturnValue({
      markMovieAsWatched: mockMarkMovieAsWatched
    })
  };
  mockStore.useAuthStore.mockMarkMovieAsWatched = mockMarkMovieAsWatched;
  return mockStore;
});

jest.mock('axios', () => ({
  post: jest.fn().mockResolvedValue({ data: {} }),
  get: jest.fn().mockResolvedValue({ data: {} }),
}));

jest.mock('../../components/Navbar', () => () => <div data-testid="navbar-mock">Navbar</div>);

jest.mock('../../components/RecommendationCard', () => ({ 
  movie, 
  explanation, 
  trailer, 
  isInWatchlist, 
  addToWatchlist,
  openRatingModal 
}) => {
  const mockEvent = {
    currentTarget: {
      getBoundingClientRect: () => ({
        top: 0,
        left: 0
      })
    }
  };
  
  return (
    <div data-testid={`movie-card-${movie.id}`} className="recommendation-card-mock">
      <h3>{movie.title}</h3>
      <p>{explanation}</p>
      <button 
        onClick={() => openRatingModal(movie.id, mockEvent)}
        aria-label="Rate this movie"
        data-testid={`rate-button-${movie.id}`}
      >
        Rate Movie
      </button>
    </div>
  );
});

jest.mock('react-hot-toast', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn()
  }
}));

describe('DashboardPage Component', () => {
  const mockMovieData = [
    {
      id: 123,
      title: 'Test Movie 1',
      poster_path: '/poster1.jpg',
      release_date: '2023-01-01',
      overview: 'Overview 1',
      genres: ['Action']
    },
    {
      id: 456,
      title: 'Test Movie 2',
      poster_path: '/poster2.jpg',
      release_date: '2023-02-01',
      overview: 'Overview 2',
      genres: ['Comedy']
    }
  ];

  const mockRecommendations = {
    recommendations: ['123', '456'],
    explanations: [
      'Recommended based on your preferences.',
      'Matches your favorite genres.'
    ]
  };

  const mockStoreData = {
    user: { 
      _id: 'user123', 
      name: 'Test User',
      watchlist: [],
      watchedMovies: []
    },
    isAuthenticated: true,
    addToWatchlist: jest.fn(),
    isInWatchlist: jest.fn().mockReturnValue(false),
    markMovieAsWatched: jest.fn().mockResolvedValue({ success: true })
  };

  beforeEach(() => {
    useAuthStore.mockReturnValue(mockStoreData);
    axios.post.mockImplementation(() => Promise.resolve({ data: mockRecommendations }));
    global.fetch = jest.fn().mockImplementation((url) => {
      const movieId = url.match(/\/movie\/(\d+)/)?.[1];
      const movie = mockMovieData.find(m => m.id.toString() === movieId);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(movie || {})
      });
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders dashboard and fetches recommendations', async () => {
    render(<DashboardPage />);
    expect(screen.getByText('Loading your personalized recommendations...')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('Recommended For You')).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByText('Test Movie 1')).toBeInTheDocument();
      expect(screen.getByText('Test Movie 2')).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByText('Recommended based on your preferences.')).toBeInTheDocument();
      expect(screen.getByText('Matches your favorite genres.')).toBeInTheDocument();
    });
  });

  test('handles case when no recommendations are available', async () => {
    axios.post.mockImplementationOnce(() => Promise.resolve({ 
      data: {
        recommendations: [],
        explanations: [],
        message: 'No recommendations available'
      }
    }));
    render(<DashboardPage />);
    await waitFor(() => {
      expect(screen.getByText('No recommendations available')).toBeInTheDocument();
    });
    expect(screen.queryByTestId(/movie-card-/)).not.toBeInTheDocument();
  });

  test('handles API error when fetching recommendations', async () => {
    axios.post.mockImplementationOnce(() => Promise.reject({ 
      response: { data: { error: 'Failed to get recommendations' } }
    }));
    render(<DashboardPage />);
    await waitFor(() => {
      expect(screen.getByText('Failed to get recommendations')).toBeInTheDocument();
    });
  });

  test('opens rating modal when rate button is clicked', async () => {
    const user = userEvent.setup();
    render(<DashboardPage />);
    await waitFor(() => {
      expect(screen.getByText('Recommended For You')).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByTestId('movie-card-123')).toBeInTheDocument();
    });
    const rateButton = screen.getByTestId('rate-button-123');
    await user.click(rateButton);
    await waitFor(() => {
      expect(screen.getByText('Rate this movie')).toBeInTheDocument();
      expect(screen.getAllByRole('button', { name: '★' })).toHaveLength(5);
    });
  });

  test('handles movie rating submission', async () => {
    const user = userEvent.setup();
    render(<DashboardPage />);
    await waitFor(() => {
      expect(screen.getByText('Recommended For You')).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByTestId('movie-card-123')).toBeInTheDocument();
    });
    const rateButton = screen.getByTestId('rate-button-123');
    await user.click(rateButton);
    await waitFor(() => {
      expect(screen.getByText('Rate this movie')).toBeInTheDocument();
    });
    const starButtons = screen.getAllByRole('button', { name: '★' });
    await user.click(starButtons[3]); 
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    const mockStore = require('../../store/authStore');
    expect(mockStore.getState().markMovieAsWatched).toHaveBeenCalledWith(123, 4);
  });
});
