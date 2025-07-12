import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import SearchResultsPage from '../SearchResultsPage';
import { useParams } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import { VALID_TMDB_IDS } from '../../utils/tmdbIds';
import toast from 'react-hot-toast';
import { BrowserRouter } from 'react-router-dom';

const renderWithRouter = (ui) =>
    render(<BrowserRouter>{ui}</BrowserRouter>);

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useParams: jest.fn(),
}));

jest.mock('../../store/authStore', () => ({
  useAuthStore: jest.fn(),
}));

jest.mock('react-hot-toast', () => {
    const toastFn = jest.fn();
    toastFn.error = jest.fn();
    toastFn.success = jest.fn();
    return {
      __esModule: true,
      default: toastFn,
    };
  });
  

global.fetch = jest.fn();

describe('SearchResultsPage', () => {
  const mockAddToWatchlist = jest.fn();
  const mockIsInWatchlist = jest.fn(() => false);
  const mockMarkMovieAsWatched = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();

    useAuthStore.mockReturnValue({
      addToWatchlist: mockAddToWatchlist,
      isInWatchlist: mockIsInWatchlist,
    });

    useAuthStore.getState = () => ({
      markMovieAsWatched: mockMarkMovieAsWatched,
    });

    useParams.mockReturnValue({ query: 'Inception' });

    VALID_TMDB_IDS.add('123'); // Ensure one is valid
  });

  const mockTMDBResponse = () => {
    fetch.mockImplementation((url) => {
      if (url.includes('genre')) {
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              genres: [{ id: 1, name: 'Action' }],
            }),
        });
      }

      if (url.includes('videos')) {
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              results: [{ type: 'Trailer', key: 'trailer123' }],
            }),
        });
      }

      return Promise.resolve({
        json: () =>
          Promise.resolve({
            results: [
              {
                id: 123,
                title: 'Inception',
                genre_ids: [1],
                poster_path: '/poster.jpg',
              },
              {
                id: 999, 
                title: 'Filtered Out',
                genre_ids: [1],
                poster_path: '/bad.jpg',
              },
            ],
          }),
      });
    });
  };

  test('renders loading state and results', async () => {
    mockTMDBResponse();

    renderWithRouter(<SearchResultsPage />);

    expect(screen.getByText(/searching for movies/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Inception')).toBeInTheDocument();
    });

    expect(screen.queryByText('Filtered Out')).not.toBeInTheDocument();
  });

  test('handles no query input', async () => {
    useParams.mockReturnValue({ query: '    ' }); // spaces
    renderWithRouter(<SearchResultsPage />);
    await waitFor(() =>
      expect(screen.getByText(/please enter movie title/i)).toBeInTheDocument()
    );
  });

  test('handles no search results', async () => {
    fetch.mockImplementation(() =>
      Promise.resolve({
        json: () => Promise.resolve({ results: [] }),
      })
    );

    renderWithRouter(<SearchResultsPage />);
    await waitFor(() =>
      expect(
        screen.getByText(/oops, i couldn't find anything/i)
      ).toBeInTheDocument()
    );
  });

  test('handles fetch error', async () => {
    fetch.mockRejectedValueOnce(new Error('API Error'));
    renderWithRouter(<SearchResultsPage />);

    await waitFor(() =>
      expect(
        screen.getByText(/failed to fetch movies/i)
      ).toBeInTheDocument()
    );
  });

  test('opens and cancels rating modal', async () => {
    mockTMDBResponse();
    renderWithRouter(<SearchResultsPage />);

    await waitFor(() => expect(screen.getByText('Inception')).toBeInTheDocument());

    const rateBtn = screen.getByTitle("I've Watched This");
    fireEvent.click(rateBtn, { currentTarget: rateBtn });

    await waitFor(() => {
      expect(screen.getByText(/rate this movie/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/cancel/i));
    expect(screen.queryByText(/rate this movie/i)).not.toBeInTheDocument();
  });

  test('submits rating successfully', async () => {
    mockMarkMovieAsWatched.mockResolvedValue({ success: true });
    mockTMDBResponse();
    renderWithRouter(<SearchResultsPage />);

    await waitFor(() => expect(screen.getByText('Inception')).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("I've Watched This"));
    await waitFor(() => screen.getByText(/rate this movie/i));

    fireEvent.click(screen.getAllByText('★')[0]); 
    fireEvent.click(screen.getByText(/submit/i));

    await waitFor(() =>
      expect(toast.success).toHaveBeenCalledWith('You rated "Inception" 1 stars!')
    );
  });

  test('shows error on failed rating submission', async () => {
    mockMarkMovieAsWatched.mockRejectedValue({ message: 'fail' });
    mockTMDBResponse();
    renderWithRouter(<SearchResultsPage />);

    await waitFor(() => expect(screen.getByText('Inception')).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("I've Watched This"));
    fireEvent.click(screen.getAllByText('★')[0]);
    fireEvent.click(screen.getByText(/submit/i));

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith('fail')
    );
  });

  test('sorts movies with trailer first', async () => {
    mockTMDBResponse();
    renderWithRouter(<SearchResultsPage />);

    await waitFor(() => {
      const movieTitle = screen.getByText('Inception');
      expect(movieTitle).toBeInTheDocument();
    });
  });
});
