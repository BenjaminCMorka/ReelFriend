module.exports = {
    ...require('./jest.config'),
    collectCoverage: true,
    coverageDirectory: 'coverage',
    coverageReporters: ['text', 'lcov', 'clover', 'html'],
    collectCoverageFrom: [
      'src/**/*.{js,jsx}',
      '!src/**/*.d.ts',
      '!src/index.js',
      '!src/reportWebVitals.js',
      '!src/serviceWorker.js',
      '!src/setupTests.js',
      '!src/testUtils.js',
      '!**/node_modules/**',
      '!**/vendor/**'
    ],
    coverageThreshold: {
      global: {
        branches: 70,
        functions: 70,
        lines: 70,
        statements: 70
      }
    }
  };