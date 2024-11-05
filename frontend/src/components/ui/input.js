// src/components/ui/input.js
import React from 'react';

const Input = React.forwardRef(({ className, ...props }, ref) => {
    return (
        <input 
            ref={ref}
            className={`border border-gray-300 rounded p-2 focus:outline-none focus:ring focus:border-blue-300 ${className}`} 
            {...props} 
        />
    );
});

export default Input;
