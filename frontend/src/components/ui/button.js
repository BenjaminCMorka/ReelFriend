// src/components/ui/button.js
import React from 'react';

const Button = ({ variant, className, children, ...props }) => {
    const baseClasses = 'py-2 px-4 rounded focus:outline-none focus:ring';
    const variantClasses = variant === 'ghost' ? 'bg-transparent text-gray-800' : 'bg-blue-500 text-white';
    
    return (
        <button className={`${baseClasses} ${variantClasses} ${className}`} {...props}>
            {children}
        </button>
    );
};

export default Button;
