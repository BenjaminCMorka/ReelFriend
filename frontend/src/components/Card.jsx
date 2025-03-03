
const Card = ({ children }) => {
    return (
        <div >
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