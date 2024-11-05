// src/components/ui/card.js
import React from 'react';

const Card = ({ className = '', children }) => {
    return (
        <div className={`border rounded-lg shadow-lg bg-white p-4 ${className}`}>
            {children}
        </div>
    );
};

const CardContent = ({ className = '', children }) => {
    return (
        <div className={`p-4 ${className}`}>
            {children}
        </div>
    );
};

export { Card, CardContent };
