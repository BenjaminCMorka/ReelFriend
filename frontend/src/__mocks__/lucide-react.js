
const createIconMock = (name) => {
    const IconComponent = (props) => (
      <svg data-testid={`${name.toLowerCase()}-icon`} {...props} />
    );
    IconComponent.displayName = name;
    return IconComponent;
  };
  

  export const Mail = createIconMock('Mail');
  export const Lock = createIconMock('Lock');
  export const User = createIconMock('User');
  export const Loader = createIconMock('Loader');
  export const ArrowLeft = createIconMock('ArrowLeft');
  export const Eye = createIconMock('Eye');
  export const EyeOff = createIconMock('EyeOff');
  export const Search = createIconMock('Search');
  export const Check = createIconMock('Check');
  export const X = createIconMock('X');
  export const Camera = createIconMock('Camera');
  export const Upload = createIconMock('Upload');
  export const ChevronDown = createIconMock('ChevronDown');
  export const ChevronUp = createIconMock('ChevronUp');
  export const ChevronLeft = createIconMock('ChevronLeft');
  export const ChevronRight = createIconMock('ChevronRight');
  export const Home = createIconMock('Home');
  export const Settings = createIconMock('Settings');
  export const LogOut = createIconMock('LogOut');
  export const Star = createIconMock('Star');
  export const Trash = createIconMock('Trash');
  export const Edit = createIconMock('Edit');
  export const Plus = createIconMock('Plus');
  export const Minus = createIconMock('Minus');
  export const Menu = createIconMock('Menu');