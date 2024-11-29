from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

class Folder(db.Model):
    __tablename__ = 'Folders'
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(127), unique=True, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'path': self.path,
        }
    
class Video(db.Model):
    __tablename__ = 'Videos'
    id = db.Column(db.Integer, primary_key=True)
    folderId = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    fps = db.Column(db.Integer, nullable=False)
    training = db.Column(db.Boolean, nullable=False)
    quality = db.Column(db.Integer, nullable=False)
    obstruction = db.Column(db.Boolean, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'folderId' : self.folderId,
            'name' : self.name,
            'width' : self.width,
            'height' : self.height,
            'fps' : self.fps,
            'training' : self.training,
            'quality' : self.quality,
            'obstruction' : self.obstruction
        }