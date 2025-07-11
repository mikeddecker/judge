from domain.folder import Folder
from domain.frameinfo import FrameInfo
from domain.job import Job
from domain.tag import Tag
from domain.tagGroup import TagGroup
from domain.skill import Skill
from domain.videoinfo import VideoInfo
from helpers.ConfigHelper import get_discipline_DoubleDutch_config
from repository.models import Folder as FolderDB, Video as VideoDB, FrameLabel, Skillinfo_DoubleDutch, Jobs as JobDB
from repository.models import Tag as TagDB, TagGroup as TagGroupDB 
from typing import List

class MapToDomain:
    def map_folder(folderDB: FolderDB) -> Folder:
        folder= Folder(folderDB.id, folderDB.name, None)
        original = folder
        while folderDB.parent is not None:
            parent_folder = Folder(folderDB.parent.id, folderDB.parent.name, None)
            folder.Parent = parent_folder
            folder = parent_folder
            folderDB = folderDB.parent
        return original
    
    # TODO : update when height, width... are required & implemented
    def map_video(videoDB: VideoDB) -> VideoInfo:
        video = VideoInfo(
            id = videoDB.id,
            name = videoDB.name,
            folder = MapToDomain.map_folder(videoDB.folder),
            frameLength = videoDB.frameLength,
            duration = videoDB.duration,
            fps = videoDB.fps,
            completed_skill_labels = videoDB.completed_skill_labels,
            width=videoDB.width,
            height=videoDB.height,
            judgeDiffScore=videoDB.judgeDiffScore
        )
        for f in videoDB.frameLabels:
            video.add_framelabel(FrameInfo(frameNr=f.frameNr, x=f.x, y=f.y, width=f.width, height=f.height, jumperVisible=f.jumperVisible, labeltype=f.labeltype))
        return video
    
    def map_skills(skillsDB: Skillinfo_DoubleDutch) -> List[Skill]:
        return [
            Skill(
                id=s.id,
                disciplineConfig=get_discipline_DoubleDutch_config(),
                skillinfo={
                    "Type" : s.type,
                    "Rotations" : s.rotations,
                    "Turner1" : s.turner1,
                    "Turner2" : s.turner2,
                    "Skill" : s.skill,
                    "Hands" : s.hands,
                    "Feet" : s.feet,
                    "Turntable" : s.turntable,
                    "BodyRotations" : s.bodyRotations,
                    "Backwards" : s.backwards,
                    "Sloppy" : s.sloppy,
                    "Hard2see" : s.hard2see,
                    "Fault" : s.fault,
                },
                start=s.frameStart,
                end=s.frameEnd
            )
            for s
            in skillsDB
        ]
    
    def map_job(jobDB: JobDB) -> Job:
        return Job(
            id = jobDB.id,
            type = jobDB.type,
            step = jobDB.step,
            job_arguments = jobDB.job_arguments,
            request_time = jobDB.request_time,
            status = jobDB.status,
            status_details = jobDB.status_details,
        )
    
    def map_tag(tagDB: TagDB) -> Tag:
        return Tag(
            id = tagDB.id,
            name = tagDB.name,
            tagGroup = None if not tagDB.tagGroupId else TagGroup(
                id = tagDB.tagGroupId,
                name = tagDB.group.name, # Add tags if necessairy
            )
        )
    
    def map_tag_group(tagGroupDB: TagGroupDB) -> TagGroup:
        return TagGroup(
            id = tagGroupDB.id,
            name = tagGroupDB.name,
            tags = [Tag(id = t.id, name = t.name) for t in tagGroupDB.tags]
        )