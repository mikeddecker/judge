from typing import Optional, Final

class Folder:
    def __init__(self, id: Optional[int], name: str, parent: Optional['Folder'] = None):
        # Initialize the instance with values (this won't change the class-level variables)
        if (id is None) or (id is not None and id <= 0):
            raise ValueError("Id must be strict positive")
        if not name:
            raise ValueError("Name may not be an empty string")
        if parent is not None and not isinstance(parent, Folder):
            raise ValueError(f"parent must be of type {Folder}, got {type(parent)}")


        self.id = id
        self.name = name
        self.parent = parent

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting 'id' after it is set in __init__
            if name in ["id", "name", "parent"]:
                raise AttributeError(f"Cannot modify '{name}' once it is set")
        super().__setattr__(name, value)
