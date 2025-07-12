
const mockAxios = {
  defaults: {
    withCredentials: false  
  },
  get: jest.fn().mockResolvedValue({ data: {} }),
  post: jest.fn().mockResolvedValue({ data: { success: true } }),
  put: jest.fn().mockResolvedValue({ data: { success: true } }),
  delete: jest.fn().mockResolvedValue({ data: { success: true } }),
  create: jest.fn().mockReturnThis(),
};

module.exports = mockAxios;