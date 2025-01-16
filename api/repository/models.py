# from flask_sqlalchemy import SQLAlchemy
from repository.db import db
from sqlalchemy.dialects.mysql import SMALLINT

class Folder(db.Model):
    __tablename__ = 'Folders'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(127), nullable=False)
    parentId = db.Column(db.Integer, db.ForeignKey('Folders.id'), nullable=True)
    parent = db.relationship('Folder', remote_side=[id], backref='children', lazy='joined')
    videos = db.relationship('Video', backref='folder', lazy='dynamic') # Loaded lazily, so videoIDs are accecible, but full fetch only when explicitly asked

    # Define a composite unique constraint
    __table_args__ = (
        db.UniqueConstraint('name', 'parentId', name='_name_parent_unique_constraint'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'parentId' : self.parentId,
            'children': [child.id for child in self.children],
            'videoIds': [video.id for video in self.videos] 
        }
    
class Source(db.Model):
    __tablename__ = 'Sources'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(127), nullable=False)


    
class Video(db.Model):
    __tablename__ = 'Videos'
    id = db.Column(db.Integer, primary_key=True)
    folderId = db.Column(db.Integer, db.ForeignKey('Folders.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    frameLength = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    fps = db.Column(db.Float, nullable=False)
    training = db.Column(db.Boolean, nullable=False)
    qualitative = db.Column(db.Boolean, nullable=False)
    obstruction = db.Column(db.Boolean, nullable=False)
    private = db.Column(db.Boolean, nullable=False, default=False)
    source = db.Column(db.Integer, nullable=True)
    sourceInfo = db.Column(db.String(255), nullable=True)

    frameLabels = db.relationship('FrameLabel', backref='video', lazy='joined')

    __table_args__ = (
        db.UniqueConstraint('name', 'folderId', name='_name_folder_unique_constraint'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'folderId' : self.folderId,
            'name' : self.name,
            'frameLength': self.frameLength,
            'width' : self.width,
            'height' : self.height,
            'fps' : self.fps,
            'training' : self.training,
            'qualitative' : self.qualitative,
            'obstruction' : self.obstruction
        }
    
class FrameLabel(db.Model):
    __tablename__ = 'FrameLabels'
    videoId = db.Column(db.Integer, db.ForeignKey('Videos.id'), nullable=False)
    frameNr = db.Column(SMALLINT(unsigned=True), nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    jumperVisible = db.Column(db.Boolean, nullable=False, default=True)

    # Define a composite unique constraint
    __table_args__ = (
        db.PrimaryKeyConstraint('videoId', 'frameNr'),
    )

    def to_dict(self):
        return {
            'videoId' : self.videoId,
            'frameNr' : self.frameNr,
            'x' : self.x,
            'y' : self.y,
            'width' : self.width,
            'jumperVisible' : self.jumperVisible
        }
    