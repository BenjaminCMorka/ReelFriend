const SearchInput = ({ className, ...props }) => {
    return (
        <input 
            className={`border border-gray-300 rounded p-2 focus:outline-none focus:ring focus:border-blue-300 ${className}`} 
            {...props} 
        />
    );
};

export default SearchInput;
