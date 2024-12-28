from helpers.ValueHelper import ValueHelper

class FrameInfo:
    PROPERTIES = ["FrameNr", "X", "Y", "Width", "Height", "JumperVisible"]
    # Frame does not 

    def __init__(self, frameNr: int, x: int, y: int, width: int, height: int, jumperVisible: bool = True):
        self.__check_raise_values(x=x, y=y, width=width, height=height, jumperVisible=jumperVisible)
        self.__setFrameNr(frameNr)
        self.setX(x)
        self.setY(y)
        self.setWidth(width)
        self.setHeight(height)
        self.setJumperVisible(jumperVisible)

    def __setattr__(self, name, value):
        # Prevent setting immutable attributes after it is set in __init__
        if name == 'FrameNr':
            self.__setFrameNr(value)
        if name == 'X':
            self.setX(value)
        if name == 'Y':
            self.setY(value)
        if name == "Width":
            self.setWidth(value)
        if name == "Heigth":
            self.setHeight(value)
        if name == "JumerVisible":
            self.setJumperVisible(value)
        if name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def __setFrameNr(self, framenr: int):
        ValueHelper.check_raise_frameNr(framenr)
        if hasattr(self, 'FrameNr') and self.FrameNr is not None:
            raise AttributeError(f"Cannot modify FrameNr once it is set")
        if framenr is None or framenr < 0:
            raise ValueError("FrameNr must be strict positive")
        object.__setattr__(self, 'FrameNr', framenr)

    def setX(self, x: float):
        if x < 0 or x > 1.0:
            raise ValueError(f"X must be in [0..1], got {x}")
        self.__check_raise_values(
            x=x, 
            y=0.5 if not hasattr(self, "Y") else self.Y, 
            width=0.5 if not hasattr(self, "Width") else self.Width, 
            height=0.5 if not hasattr(self, "Height") else self.Height, 
            jumperVisible=True if not hasattr(self, "JumperVisible") else self.JumperVisible)
        object.__setattr__(self, 'X', x)
    
    def setY(self, y: float):
        if y < 0 or y > 1.0:
            raise ValueError(f"Y must be in [0..1], got {y}")
        self.__check_raise_values(
            x=0.5 if not hasattr(self, "X") else self.X,
            y=y,
            width=0.5 if not hasattr(self, "Width") else self.Width, 
            height=0.5 if not hasattr(self, "Height") else self.Height, 
            jumperVisible=True if not hasattr(self, "JumperVisible") else self.JumperVisible)
        object.__setattr__(self, 'Y', y)

    def setWidth(self, width: float):
        if width < 0 or width > 1.0:
            raise ValueError(f"Width must be in [0..1], got {width}")
        self.__check_raise_values(
            x=0.5 if not hasattr(self, "X") else self.X,
            y=0.5 if not hasattr(self, "Y") else self.Y, 
            width=width,
            height=0.5 if not hasattr(self, "Height") else self.Height, 
            jumperVisible=True if not hasattr(self, "JumperVisible") else self.JumperVisible)
        object.__setattr__(self, 'Width', width)
    
    def setHeight(self, height: float):
        if height < 0 or height > 1.0:
            raise ValueError(f"Y must be in [0..1], got {height}")
        self.__check_raise_values(
            x=0.5 if not hasattr(self, "X") else self.X,
            y=0.5 if not hasattr(self, "Y") else self.Y, 
            width=0.5 if not hasattr(self, "Width") else self.Width, 
            height=height, 
            jumperVisible=True if not hasattr(self, "JumperVisible") else self.JumperVisible)
        object.__setattr__(self, 'Height', height)

    def setJumperVisible(self, visible: bool):
        if not isinstance(visible, bool):
            raise ValueError(f"Jumper visible must be a boolean")
        self.__check_raise_values(x=self.X, y=self.Y, width=self.Width, height=self.Height, jumperVisible=visible)
        object.__setattr__(self, 'JumperVisible', visible)

    def __check_raise_values(self, x: int, y: int, width: int, height: int, jumperVisible: bool = True):
        if not jumperVisible and (
            x != 0.5 or 
            y != 0.5 or 
            width != 1.0 or
            height != 1.0):
            raise ValueError(f"View must be 100% when jumper is not visible\nGot x={x}, y={y}, width={width}, height={height}")

    # TODO : update when more equal checks are performed metadata is extended
    def __eq__(self, value : object):
        if not isinstance(value, FrameInfo):
            raise ValueError(f"Value not a {FrameInfo} got {type(value)} instead")
        
        # Typehint
        other : FrameInfo = value
        
        return (
            self.FrameNr == other.FrameNr and
            self.X == other.X and
            self.Y == other.Y and
            self.Width == other.Width and
            self.Height == other.Height and
            self.JumperVisible == other.JumperVisible
        )

    def __str__(self):
        return f'frameNr = {self.FrameNr}, x = {self.X:.2f}, y = {self.Y:.2f}, w = {self.Width:.2f}, h = {self.Height:.2f}, visible = {self.JumperVisible}'
