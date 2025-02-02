from domain.folder import Folder
from domain.frameinfo import FrameInfo
from domain.videoinfo import VideoInfo
from repository.models import Folder as FolderDB, FrameLabel
from sqlalchemy.orm.session import Session

class MapToDB:
    def map_frameInfo(video: VideoInfo, frameInfo: FrameInfo) -> FrameLabel:
        return FrameLabel(
            videoId = video.Id,
            frameNr = frameInfo.FrameNr,
            x = frameInfo.X,
            y = frameInfo.Y,
            width = frameInfo.Width,
            height = frameInfo.Height,
            jumperVisible = frameInfo.JumperVisible,
            labeltype = frameInfo.LabelType
        )
