import axios from 'axios';
import { act } from 'react';
import { useAuthStore } from '../authStore';


jest.mock('axios');

describe('authStore', () => {
  afterEach(() => {
    jest.clearAllMocks();
    act(() => {
      useAuthStore.setState({
        isAuthenticated: false,
        user: null,
        isLoading: false,
        error: null,
      });
    });
  });

  test('initial state', () => {
    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
    expect(state.user).toBe(null);
    expect(state.isLoading).toBe(false);
    expect(state.error).toBe(null);
  });

  test('login success', async () => {
    const mockUser = { id: '1', name: 'Ben' };
    axios.post.mockResolvedValueOnce({
      data: { user: mockUser }
    });

    await act(async () => {
      await useAuthStore.getState().login('test@example.com', 'pass123');
    });

    const state = useAuthStore.getState();
    expect(state.user).toEqual(mockUser);
    expect(state.isAuthenticated).toBe(true);
    expect(state.error).toBe(null);
  });

  test('login failure', async () => {
    const errorMessage = 'Invalid credentials';
    axios.post.mockRejectedValueOnce({
      response: { data: { message: errorMessage } }
    });

    await act(async () => {
      try {
        await useAuthStore.getState().login('bad@example.com', 'wrong');
      } catch (error) {
        // Expected to throw error
      }
    });

    const state = useAuthStore.getState();
    expect(state.user).toBe(null);
    expect(state.isAuthenticated).toBe(false);
    expect(state.error).toBe(errorMessage);
  });

  test('signup success', async () => {
    const mockUser = { id: '2', name: 'Alice' };
    axios.post.mockResolvedValueOnce({
      data: { user: mockUser }
    });

    await act(async () => {
      await useAuthStore.getState().signup('a@example.com', '123', 'Alice');
    });

    const state = useAuthStore.getState();
    expect(state.user).toEqual(mockUser);
    expect(state.isAuthenticated).toBe(true);
    expect(state.error).toBe(null);
  });

  test('signup failure', async () => {
    const errorMessage = 'Email exists';
    axios.post.mockRejectedValueOnce({
      response: { data: { message: errorMessage } }
    });

    await act(async () => {
      try {
        await useAuthStore.getState().signup('a@example.com', '123', 'Alice');
      } catch (error) {
        // Expected to throw error
      }
    });

    const state = useAuthStore.getState();
    expect(state.user).toBe(null);
    expect(state.isAuthenticated).toBe(false);
    expect(state.error).toBe(errorMessage);
  });

  test('logout clears state', async () => {
    axios.post.mockResolvedValueOnce({});


    useAuthStore.setState({
      user: { id: '1' },
      isAuthenticated: true,
    });

    await act(async () => {
      await useAuthStore.getState().logout();
    });

    const state = useAuthStore.getState();
    expect(state.user).toBe(null);
    expect(state.isAuthenticated).toBe(false);
  });

  test('resetPassword success', async () => {
    const successMessage = 'Password reset successfully';
    axios.post.mockResolvedValueOnce({
      data: { message: successMessage }
    });

    await act(async () => {
      await useAuthStore.getState().resetPassword('test@example.com', 'newpass');
    });

    const state = useAuthStore.getState();
    expect(state.error).toBe(null);
    expect(state.message).toBe(successMessage);
  });

  test('resetPassword failure', async () => {
    const errorMessage = 'Reset failed';
    axios.post.mockRejectedValueOnce({
      response: { data: { message: errorMessage } }
    });

    await act(async () => {
      try {
        await useAuthStore.getState().resetPassword('test@example.com', 'newpass');
      } catch (error) {
        // Expected to throw error
      }
    });

    expect(useAuthStore.getState().error).toBe(errorMessage);
  });

  test('markMovieAsWatched updates user', async () => {
    const mockUser = { 
      id: '1', 
      name: 'Ben',
      watched: [] 
    };
    axios.post.mockResolvedValueOnce({
      data: { 
        success: true,
        user: { 
          ...mockUser, 
          watched: ['123'] 
        } 
      }
    });

    useAuthStore.setState({ user: mockUser });

    await act(async () => {
      await useAuthStore.getState().markMovieAsWatched('123', 4);
    });

    const state = useAuthStore.getState();
    expect(state.user?.watched).toContain('123');
  });

  test('markMovieAsWatched handles failure', async () => {
    const errorMessage = 'Rating failed';
    axios.post.mockRejectedValueOnce({
      response: { 
        data: { message: errorMessage } 
      }
    });

    useAuthStore.setState({ user: { id: '1' } });

    await act(async () => {
      try {
        await useAuthStore.getState().markMovieAsWatched('999', 2);
      } catch (err) {
        expect(err.message).toBe(errorMessage);
      }
    });

    const state = useAuthStore.getState();
    expect(state.error).toBe(errorMessage);
    expect(state.isLoading).toBe(false);
  });

});