import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RecommendationCard from '../../components/RecommendationCard';
import { toast } from 'react-hot-toast';

// Mock react-hot-toast
jest.mock('react-hot-toast', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn()
  }
}));

describe('RecommendationCard', () => {
  // Sample movie data for testing
  const mockMovie = {
    id: 123,
    title: 'Test Movie',
    release_date: '2022-01-01',
    poster_path: '/test-poster.jpg',
    overview: 'This is a test movie description.',
    genres: [{ name: 'Action' }, { name: 'Adventure' }]
  };

  const mockTrailer = 'https://example.com/trailer';
  const mockExplanation = 'Recommended because you liked similar movies';
  const mockIsInWatchlist = jest.fn().mockReturnValue(false);
  const mockAddToWatchlist = jest.fn().mockResolvedValue({});
  const mockOpenRatingModal = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  const renderCard = (props = {}) => {
    return render(
      <RecommendationCard
        movie={mockMovie}
        explanation={mockExplanation}
        trailer={mockTrailer}
        isInWatchlist={mockIsInWatchlist}
        addToWatchlist={mockAddToWatchlist}
        openRatingModal={mockOpenRatingModal}
        {...props}
      />
    );
  };

  it('renders the card with movie information', () => {
    // Act
    renderCard();
    
    // Assert
    expect(screen.getByText('Test Movie')).toBeInTheDocument();
    expect(screen.getByText('2022')).toBeInTheDocument();
    expect(screen.getByText('Action, Adventure')).toBeInTheDocument();
    expect(screen.getByRole('img')).toHaveAttribute('src', 'https://image.tmdb.org/t/p/w200/test-poster.jpg');
    expect(screen.getByRole('img')).toHaveAttribute('alt', 'Test Movie');
  });

  it('renders trailer button when trailer is available', () => {
    // Act
    renderCard();
    
    // Assert
    const trailerLink = screen.getByRole('link');
    expect(trailerLink).toHaveAttribute('href', mockTrailer);
    expect(screen.getByText('Watch Trailer')).toBeInTheDocument();
  });

  it('shows no trailer available message when trailer is not provided', () => {
    // Act
    renderCard({ trailer: null });
    
    // Assert - need to find a way to check the "no trailer" text that appears on hover
    expect(screen.queryByText('Watch Trailer')).not.toBeInTheDocument();
    expect(screen.getByText('No Trailer Available')).toBeInTheDocument();
  });

  it('adds movie to watchlist when add to watchlist button is clicked', async () => {
    // Arrange
    renderCard();
    
    // Act
    const addToWatchlistButton = screen.getAllByRole('button')[0]; // First button is add to watchlist
    fireEvent.click(addToWatchlistButton);
    
    // Assert
    expect(mockAddToWatchlist).toHaveBeenCalledWith(
      '123', 
      'Test Movie',
      '/test-poster.jpg'
    );
    await waitFor(() => {
      expect(toast.success).toHaveBeenCalledWith('Added "Test Movie" to your watchlist!');
    });
  });

  it('shows error toast if movie is already in watchlist', async () => {
    // Arrange
    mockIsInWatchlist.mockReturnValueOnce(true);
    renderCard();
    
    // Act
    const addToWatchlistButton = screen.getAllByRole('button')[0];
    fireEvent.click(addToWatchlistButton);
    
    // Assert
    expect(mockAddToWatchlist).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('"Test Movie" is already in your watchlist!');
    });
  });

  it('shows error toast if addToWatchlist fails', async () => {
    // Arrange
    mockAddToWatchlist.mockRejectedValueOnce(new Error('Failed to add'));
    renderCard();
    
    // Act
    const addToWatchlistButton = screen.getAllByRole('button')[0];
    fireEvent.click(addToWatchlistButton);
    
    // Assert
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Failed to add to watchlist. Please try again.');
    });
  });

  it('opens rating modal when watch button is clicked', () => {
    // Arrange
    renderCard();
    
    // Act
    const watchButton = screen.getAllByRole('button')[1]; // Second button is mark as watched
    fireEvent.click(watchButton);
    
    // Assert
    expect(mockOpenRatingModal).toHaveBeenCalledWith(123, expect.anything());
  });

  it('toggles movie description when description button is clicked', () => {
    // Arrange
    renderCard();
    
    // Act - initially description should not be visible
    expect(screen.queryByText('This is a test movie description.')).not.toBeInTheDocument();
    
    // Click description toggle button
    const descriptionButton = screen.getAllByRole('button')[2]; // Third button is description toggle
    fireEvent.click(descriptionButton);
    
    // Assert - description should now be visible
    expect(screen.getByText('This is a test movie description.')).toBeInTheDocument();
    
    // Act - click again to hide
    fireEvent.click(descriptionButton);
    
    // Assert - description should be hidden again
    expect(screen.queryByText('This is a test movie description.')).not.toBeInTheDocument();
  });

  it('toggles explanation when explanation button is clicked', () => {
    // Arrange
    renderCard();
    
    // Act - initially explanation should not be visible
    expect(screen.queryByText(mockExplanation)).not.toBeInTheDocument();
    
    // Click explanation toggle button
    const explanationButton = screen.getAllByRole('button')[3]; // Fourth button is explanation toggle
    fireEvent.click(explanationButton);
    
    // Assert - explanation should now be visible
    expect(screen.getByText(mockExplanation)).toBeInTheDocument();
    
    // Act - click again to hide
    fireEvent.click(explanationButton);
    
    // Assert - explanation should be hidden again
    expect(screen.queryByText(mockExplanation)).not.toBeInTheDocument();
  });

  it('handles missing movie data gracefully', () => {
    // Arrange & Act
    const { container } = render(<RecommendationCard />);
    
    // Assert
    expect(container).toBeEmptyDOMElement();
  });

  it('displays unknown year if release date is missing', () => {
    // Arrange
    const movieWithoutDate = {
      ...mockMovie,
      release_date: null
    };
    
    // Act
    renderCard({ movie: movieWithoutDate });
    
    // Assert
    expect(screen.getByText('Unknown')).toBeInTheDocument();
  });

  it('handles genres as a string instead of array', () => {
    // Arrange
    const movieWithGenreString = {
      ...mockMovie,
      genres: 'Action, Adventure'
    };
    
    // Act
    renderCard({ movie: movieWithGenreString });
    
    // Assert
    expect(screen.getByText('Action, Adventure')).toBeInTheDocument();
  });

  it('renders unknown if genres are missing', () => {
    // Arrange
    const movieWithoutGenres = {
      ...mockMovie,
      genres: null
    };
    
    // Act
    renderCard({ movie: movieWithoutGenres });
    
    // Assert
    expect(screen.getByText('Unknown')).toBeInTheDocument();
  });

  it('does not show toast if addToWatchlist is not provided', async () => {
    // Arrange
    renderCard({ addToWatchlist: null });
    
    // Act
    const addToWatchlistButton = screen.getAllByRole('button')[0];
    fireEvent.click(addToWatchlistButton);
    
    // Assert
    expect(toast.error).toHaveBeenCalledWith('Movie information is not available');
  });
});