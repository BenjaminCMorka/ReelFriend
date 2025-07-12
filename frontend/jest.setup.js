import '@testing-library/jest-dom';
import { TextEncoder, TextDecoder } from 'util';
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;


Object.defineProperty(window, 'scrollY', { value: 0, writable: true });
Object.defineProperty(window, 'scrollX', { value: 0, writable: true });


Element.prototype.getBoundingClientRect = jest.fn(() => ({
  width: 0,
  height: 0,
  top: 0,
  left: 0,
  bottom: 0,
  right: 0,
}));


window.fs = {
  readFile: jest.fn().mockResolvedValue(new Uint8Array())
};


const originalConsoleError = console.error;
console.error = (...args) => {

  if (
    typeof args[0] === 'string' && 
    (args[0].includes('Error fetching dashboard recommendations') ||
     args[0].includes('Error fetching trailer') ||
     args[0].includes('Full error in rating submission'))
  ) {
    return;
  }
  originalConsoleError(...args);
};