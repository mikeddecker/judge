from helpers.ValueHelper import ValueHelper
class Skill:
    PROPERTIES = [
        'Id', 'DottedName',
        'FrameStart', 'FrameEnd'
    ]
    FrameStart: int
    FrameEnd: int # Included

    def __init__(self, id: int, dottedName: str, frameStart: int = None, frameEnd: int = None):
        self.__setId(id)
        self.__setDottedName(dottedName)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            if name == 'Id':
                self.__setId(value)
            if name == 'DottedName':
                self.__setDottedName(value)
            # Prevent setting immutable attributes after it is set in __init__
            if name in ["DottedName", "Folder"]:
                raise AttributeError(f"Cannot modify '{name}' once it is set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def __setId(self, id: int):
        ValueHelper.check_raise_id(id)
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        if id is None or id <= 0:
            raise ValueError("Id must be strict positive")
        self.Id = id

    def __setDottedName(self, dottedName:str):
        if hasattr(self, 'DottedName') and self.DottedName is not None:
            raise AttributeError(f"Cannot modify DottedName once it is set")
        if not dottedName or dottedName.isspace():
            raise ValueError("DottedName may not be an empty string")
        self.DottedName = dottedName

    def __str__(self):
        return str(self.to_dict())
    
    def to_dict(self):
        return {
            'Id' : self.Id,
            'DottedName' : self.DottedName,
            'FrameStart' : self.FrameStart,
            'FrameEnd' : self.FrameEnd,
        }