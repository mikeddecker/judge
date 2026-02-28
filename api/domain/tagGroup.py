from uuid import UUID

class TagGroup:
    def __init__(
            self,
            id: UUID,
            name: str,
            tags: list = [],
        ):
        self.Id = id
        self.Name = name
        self.Tags = tags
    
    def to_dict(self):
        return {
            'Id' : self.Id,
            'Name' : self.Name,
            'Tags': [t.to_dict() for t in self.Tags]
        }

