import re
from .tagGroup import TagGroup
class Tag:
    def __init__(
            self,
            id: int,
            name: str,
            tagGroup: TagGroup = None,
            keywords: str = None,
        ):
        self.Id = id
        self.Name = name
        self.TagGroup = tagGroup
        try:
            self.Keywords = None
            self.Keywords : list[str] = keywords.split(',')
        except Exception:
            print(self)
            raise

    def to_dict(self):
        return {
            'Id' : self.Id.hex(),
            'Name' : self.Name,
            'TagGroup': self.TagGroup.to_dict() if self.TagGroup else None,
            'Keywords': self.Keywords,
        }
    
    def contains_keyword(self, word):
        reg_8digits = re.compile(r"^\d{8}$")
        if reg_8digits.match(word):
            # Potentially year in it
            word = word[:4]

        return word in self.Keywords 

    def __str__(self):
        return f"{self.__class__.__name__} - {self.Id} - {self.Name} - {self.Keywords}"

    def __repr__(self):
        return f"Tag(id={self.Id}, name={self.Name}, keywords={self.Keywords})"

