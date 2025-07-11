from .tagGroup import TagGroup

class Tag:
    def __init__(
            self,
            id: int,
            name: str,
            tagGroup: TagGroup = None,
        ):
        self.Id = id
        self.Name = name
        self.TagGroup = tagGroup
    
    def to_dict(self):
        return {
            'Id' : self.Id,
            'Name' : self.Name,
            'TagGroup': self.TagGroup.to_dict() if self.TagGroup else None
        }