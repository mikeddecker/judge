# from flask_sqlalchemy import SQLAlchemy
from repository.db import db
from sqlalchemy.dialects.mysql import TINYINT, SMALLINT, JSON
from sqlalchemy.ext.mutable import MutableDict
from datetime import datetime, date

class Folder(db.Model):
    __tablename__ = 'Folders'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
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

class CompetitionInfo(db.Model):
    __tablename__ = 'CompetitionInfo'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    info = db.Column(db.String(255), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    
class Video(db.Model):
    __tablename__ = 'Videos'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
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
    completed_skill_labels = db.Column(db.Boolean, nullable=False, default=False)
    competition = db.Column(db.Integer, db.ForeignKey('CompetitionInfo.id'))
    judgeDiffScore = db.Column(db.Float, nullable=True)

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
    
class FrameLabelType(db.Model):
    __tablename__ = 'FrameLabelTypes'
    id = db.Column(db.Integer, primary_key=True)
    info = db.Column(db.String(127))
    
class FrameLabel(db.Model):
    __tablename__ = 'FrameLabels'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    videoId = db.Column(db.Integer, db.ForeignKey('Videos.id'), nullable=False)
    frameNr = db.Column(SMALLINT(unsigned=True), nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    jumperVisible = db.Column(db.Boolean, nullable=False, default=True)
    labeltype = db.Column(db.Integer, db.ForeignKey('FrameLabelTypes.id'), nullable=False, default=1)
    labeldate = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'videoId' : self.videoId,
            'frameNr' : self.frameNr,
            'x' : self.x,
            'y' : self.y,
            'width' : self.width,
            'jumperVisible' : self.jumperVisible
        }
    
class TrainResult(db.Model):
    __tablename__ = 'TrainResults'
    modelname = db.Column(db.String(127), nullable=False)
    train_date = db.Column(db.Integer, nullable=False)
    epoch = db.Column(SMALLINT(unsigned=True), nullable=False)
    iou = db.Column(db.Float, nullable=False)
    loss = db.Column(db.Float, nullable=False)
    val_iou = db.Column(db.Float, nullable=False)
    val_loss = db.Column(db.Float, nullable=False)

    __table_args__ = (
        db.PrimaryKeyConstraint('modelname', 'epoch', name='_modelname_epoch_pk_constraint'),
    )

class TrainResultSkill(db.Model):
    __tablename__ = 'TrainResultsSkills'
    modelname = db.Column(db.String(127), nullable=False)
    train_date = db.Column(db.Integer, nullable=False)
    epoch = db.Column(SMALLINT(unsigned=True), nullable=False)
    loss = db.Column(db.Float, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    val_loss = db.Column(db.Float, nullable=False)
    val_accuracy = db.Column(db.Float, nullable=False)
    losses_and_metrics = db.Column(MutableDict.as_mutable(JSON), nullable=False)

    __table_args__ = (
        db.PrimaryKeyConstraint('modelname', 'epoch', name='_modelname_epoch_pk_constraint'),
    )

class Skillinfo_DoubleDutch_Type(db.Model):
    __tablename__ = 'Skillinfo_DoubleDutch_Type'
    id = db.Column(TINYINT(unsigned=True), primary_key=True, autoincrement=True)
    name = db.Column(db.String(127), nullable=False)

class Skillinfo_DoubleDutch_Turner(db.Model):
    __tablename__ = 'Skillinfo_DoubleDutch_Turner'
    id = db.Column(TINYINT(unsigned=True), primary_key=True, autoincrement=True)
    name = db.Column(db.String(127), nullable=False)
    level_dd = db.Column(TINYINT(unsigned=True), nullable=False, default="2")
    level_cw = db.Column(TINYINT(unsigned=True), nullable=False, default="2")
    cw_requires_both = db.Column(db.Boolean, nullable=False, default=False)
    dd_requires_both = db.Column(db.Boolean, nullable=False, default=False)

class Skillinfo_DoubleDutch_Skill(db.Model):
    __tablename__ = 'Skillinfo_DoubleDutch_Skill'
    id = db.Column(TINYINT(unsigned=True), primary_key=True, autoincrement=True)
    name = db.Column(db.String(127), nullable=False)
    level_dd = db.Column(db.String(8), nullable=False, default="2")
    level_cw = db.Column(db.String(8), nullable=False, default="2")

class Skillinfo_DoubleDutch(db.Model):
    __tablename__ = 'Skillinfo_DoubleDutch'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    videoId = db.Column(db.Integer, db.ForeignKey('Videos.id'), nullable=False)
    frameStart = db.Column(db.Integer, nullable=False)
    frameEnd = db.Column(db.Integer, nullable=False)
    type = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Type.id'), nullable=False)
    rotations = db.Column(TINYINT(unsigned=True), nullable=False)
    turner1 = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Turner.id'), nullable=False)
    turner2 = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Turner.id'), nullable=False)
    skill = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Skill.id'), nullable=False)
    hands = db.Column(TINYINT(unsigned=True), nullable=False)
    feet = db.Column(TINYINT(unsigned=True), nullable=False)
    turntable = db.Column(TINYINT(unsigned=True), nullable=False)
    bodyRotations = db.Column(TINYINT(unsigned=True), nullable=False)
    backwards = db.Column(db.Boolean, nullable=False)
    sloppy = db.Column(db.Boolean, nullable=False)
    hard2see = db.Column(db.Boolean, nullable=False, default=False)
    fault = db.Column(db.Boolean, nullable=False, default=False)
    labeldate = db.Column(db.DateTime, default=datetime.now)

class Prediction_Frames(db.Model):
    __tablename__ = 'Predictions_Localization'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    videoId = db.Column(db.Integer, db.ForeignKey('Videos.id'), nullable=False)
    frameNr = db.Column(db.Integer, nullable=False)
    
    x1 = db.Column(db.Integer, nullable=False)
    y1 = db.Column(db.Integer, nullable=False)
    x2 = db.Column(db.Integer, nullable=False)
    y2 = db.Column(db.Integer, nullable=False)
    locationChecked = db.Column(db.Boolean, nullable=False, default=False)
    
    segmentSplitValue = db.Column(db.Float, nullable=True)
    segmentsChecked = db.Column(db.Boolean, nullable=False, default=False)
    
    predictionDate = db.Column(db.DateTime, default=date.today())

class ML_Models(db.Model):
    __tablename__ = 'ML_Models'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type = db.Column(db.String(30), nullable=False)
    name = db.Column(db.String(127), nullable=False)

class Jobs(db.Model):
    __tablename__ = 'Jobs'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type = db.Column(db.String(30), nullable=False)
    step = db.Column(db.String(127), nullable=False)
    job_arguments = db.Column(MutableDict.as_mutable(JSON), nullable=False)
    request_time = db.Column(db.DateTime, default=datetime.now)
    status = db.Column(db.String(30), nullable=False)
    status_details = db.Column(db.String(127))


# class Skillinfo_DoubleDutch(db.Model):
#     __tablename__ = 'Predictions_SkillSegment'
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     videoId = db.Column(db.Integer, db.ForeignKey('Videos.id'), nullable=False)
#     frameStart = db.Column(db.Integer, nullable=False)
#     frameEnd = db.Column(db.Integer, nullable=False)
#     type = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Type.id'), nullable=False)
#     rotations = db.Column(TINYINT(unsigned=True), nullable=False)
#     turner1 = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Turner.id'), nullable=False)
#     turner2 = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Turner.id'), nullable=False)
#     skill = db.Column(TINYINT(unsigned=True), db.ForeignKey('Skillinfo_DoubleDutch_Skill.id'), nullable=False)
#     hands = db.Column(TINYINT(unsigned=True), nullable=False)
#     feet = db.Column(TINYINT(unsigned=True), nullable=False)
#     turntable = db.Column(TINYINT(unsigned=True), nullable=False)
#     bodyRotations = db.Column(TINYINT(unsigned=True), nullable=False)
#     backwards = db.Column(db.Boolean, nullable=False)
#     sloppy = db.Column(db.Boolean, nullable=False)
#     hard2see = db.Column(db.Boolean, nullable=False, default=False)
#     fault = db.Column(db.Boolean, nullable=False, default=False)
#     predictionDate = db.Column(db.DateTime, default=date.today())