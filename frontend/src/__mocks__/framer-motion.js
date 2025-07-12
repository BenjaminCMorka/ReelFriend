
const motion = new Proxy({}, {
    get: (_, prop) => {
      return function MockComponent({ children, ...props }) {
        const Component = prop === 'div' ? 'div' : prop;
        return <Component {...props}>{children}</Component>;
      };
    }
  });
  
  export { motion };
  export const AnimatePresence = ({ children }) => <>{children}</>;