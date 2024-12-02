from typing import Optional, Final

class Folder:
    def __init__(self, id: Optional[int], name: str, parent: Optional['Folder'] = None):
        # Initialize the instance with values (this won't change the class-level variables)
        self.__setId(id)
        if not name or name.isspace():
            raise ValueError("Name may not be an empty string")
        if parent is not None and not isinstance(parent, Folder):
            raise ValueError(f"parent must be of type {Folder}, got {type(parent)}")
        self.Name = name
        self.Parent = parent

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name == 'Id':
                self.__setId(value)
            if name in ["Name", "Parent"]:
                raise AttributeError(f"Cannot modify '{name}' once it is set")
        super().__setattr__(name, value)

    def __setId(self, id):
        if hasattr(self, 'Id') and self.Id is not None:
            raise AttributeError(f"Cannot modify Id once it is set")
        if id is None or id <= 0:
            raise ValueError("Id must be strict positive")
        self.Id = id

