from domain.folder import Folder
from domain.frameinfo import FrameInfo
from domain.job import Job
from domain.layerComposition import LayerComposition
from domain.tag import Tag
from domain.tagGroup import TagGroup
from domain.skill import Skill
from domain.videoinfo import VideoInfo
from repository.models import Folder as FolderDB, Video as VideoDB, Skill as SkillDB, Jobs as JobDB
from repository.models import Tag as TagDB, TagGroup as TagGroupDB, LayerComposition as LayerCompositionDB

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
    
    def map_skill(s: SkillDB) -> Skill:
        return Skill(
            id=s.id,
            skillinfo=dict(s.skillinfo),
            start=s.frameStart,
            end=s.frameEnd
        )
    
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
            keywords= tagDB.keywords,
            tagGroup = None if not tagDB.tagGroupId else TagGroup(
                id = tagDB.tagGroupId,
                name = tagDB.group.name, # Add tags if necessairy
            )
        )
    
    def map_tag_group(tagGroupDB: TagGroupDB) -> TagGroup:
        return TagGroup(
            id = tagGroupDB.id,
            name = tagGroupDB.name,
            tags = [Tag(id = t.id, name = t.name, keywords=t.keywords) for t in tagGroupDB.tags]
        )
    
    def map_layercomposition(compositionValuesDB: list[LayerCompositionDB]) -> LayerComposition:
        if len(compositionValuesDB) == 0:
            return None
        
        genProps = dict()
        startProps = dict()
        endProps = dict()
        stageProps = dict()
        for cDB in compositionValuesDB:
            cName = cDB.layer.name
            c = cDB.to_dict()
            match cDB.stage:
                case None:
                    genProps[cName] = c
                case 0:
                    startProps[cName] = c
                case -1:
                    endProps[cName] = c
                case _:
                    if cDB.stage in stageProps.keys():
                        stageProps[cDB.stage][cName] = c
                    else:
                        stageProps[cDB.stage] = { cName: c }

        return LayerComposition(
            compositionName=compositionValuesDB[0].compositionName,
            generalProperties=genProps,
            startProperties=startProps,
            endProperties=endProps,
            stageProperties=stageProps
        )

