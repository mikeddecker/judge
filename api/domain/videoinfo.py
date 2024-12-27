from typing import Dict, Set

from .folder import Folder
from .frameinfo import Frame
from .skill import Skill
from helpers.ValueHelper import ValueHelper

class VideoInfo:
    PROPERTIES = ["Frames", "Skills", "Id", "Name", "Folder"]
    # Frame does not 
    Frames = Dict[int, Frame] # Key = frameId, value is Frame
    Skills = Set[Skill]

    def __init__(self, id: int, name: str, folder: Folder):
        self.Frames: self.Frames = {}  # Initialize frames as an empty dictionary
        self.Skills: self.Skills = set()  # Initialize skills as an empty set

        self.__setId(id)
        if not name or name.isspace():
            raise ValueError("Name may not be an empty string")
        if folder is not None and not isinstance(folder, Folder):
            raise ValueError(f"parent must be of type {Folder}, got {type(folder)}")
        self.Name = name
        self.Folder = folder

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name == 'Id':
                self.__setId(value)
            if name in ["Name", "Folder"]:
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

    # Section : Frame functions

    # Section : Skill functions

    def add_skill(self, skill: Skill):
        if skill is None:
            raise ValueError("Skill may not be None")
        if skill in self.Skills:
            raise ValueError(f"Skill {skill} is already in the list.")
        self.Skills.add(skill)

