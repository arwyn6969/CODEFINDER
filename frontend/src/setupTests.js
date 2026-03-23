import '@testing-library/jest-dom';

const createMatchMedia = (query) => ({
  matches: false,
  media: query,
  onchange: null,
  addListener() {},
  removeListener() {},
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent() {
    return true;
  },
});

Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  writable: true,
  value: (query) => createMatchMedia(query),
});

global.matchMedia = window.matchMedia;
globalThis.matchMedia = window.matchMedia;

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = ResizeObserverMock;
global.ResizeObserver = ResizeObserverMock;
window.scrollTo = jest.fn();
