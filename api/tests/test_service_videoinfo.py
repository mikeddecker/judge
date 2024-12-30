import os

from parameterized import parameterized
from domain.folder import Folder
from domain.frameinfo import FrameInfo
from domain.videoinfo import VideoInfo
from dotenv import load_dotenv
from flask import Flask, current_app
from flask_migrate import Migrate
from flask_testing import TestCase
from repository.db import db
from repository.models import Folder as FolderDB
from services.folderService import FolderService
from services.videoService import VideoService
from tests.TestHelper import TestHelper
from typing import List

load_dotenv()
STORAGE_DIR_TEST = os.getenv("STORAGE_DIR_TEST") 
if os.path.exists(STORAGE_DIR_TEST):
    os.system(f"rm -rf {STORAGE_DIR_TEST}/*")
else:
    os.mkdir(STORAGE_DIR_TEST)

# TODO : update tests when width, height... are required

class VideoServiceTest(TestCase):
    some_folder = None
    some_nested_folder = None
    vidname = None
    
    ##################################
    # Preparing each test
    ##################################
    def create_app(self):
        app = Flask(__name__)
        app.config.from_object("config.TestConfig")
        app.json.compact = False
        db.init_app(app)
        migrate = Migrate(app, current_app)
        migrate.init_app(app, db)
        with app.app_context():
            db.create_all()
        
        self.folderService = FolderService(STORAGE_DIR_TEST)
        self.videoService = VideoService(STORAGE_DIR_TEST)


        return app
    
    def setUp(self):
        self.vidname = "vid.mp4"
        if not self.folderService.exists_path_on_drive(name="some_folder"):
            self.some_folder = self.folderService.create(name="some_folder")
            self.some_nested_folder = self.folderService.create(name="nested1", parent=None)
            self.some_nested_folder = self.folderService.create(name="nested2", parent=self.some_nested_folder)
            self.some_nested_folder = self.folderService.create(name="nested3", parent=self.some_nested_folder)
            self.some_nested_folder = self.folderService.create(name="nested4", parent=self.some_nested_folder)
            self.make_video_in_folder("vid.mp4", self.some_folder)
            self.make_video_in_folder("vid.mp4", self.some_nested_folder)
        else:
            self.some_folder = self.folderService.add_in_database(name="some_folder", parent=None)
            self.some_nested_folder = self.folderService.add_in_database(name="nested1", parent=None)
            self.some_nested_folder = self.folderService.add_in_database(name="nested2", parent=self.some_nested_folder)
            self.some_nested_folder = self.folderService.add_in_database(name="nested3", parent=self.some_nested_folder)
            self.some_nested_folder = self.folderService.add_in_database(name="nested4", parent=self.some_nested_folder)

    def tearDown(self):
        db.session.remove()
        db.drop_all()
    
    def test_setUp(self):
        assert os.path.exists(self.videoService.StorageFolder), f"Folder {self.videoService.StorageFolder} doesn't exist"
        assert self.videoService.count() == 0, "Databank not empty"

    ##################################
    # Helper functions
    ##################################
    def make_folder_in_storage_dir(self, relative_path : List[str]):
        """
        Joins the relative path, depending on the operating system, then creates the folder.
        """
        # NICE_TO_HAVE : check existence here
        os.mkdir(os.path.join(STORAGE_DIR_TEST, *relative_path))

    def make_video_in_folder(self, videoname: str, folder: Folder):
        """
        Create the video in the given folder.
        """
        with open(os.path.join(STORAGE_DIR_TEST, folder.get_relative_path(), videoname), 'w') as fp:
            pass

    ##################################
    # Test constructor
    ##################################
    def test_ctor_valid(self):
        assert os.path.exists(STORAGE_DIR_TEST), f"Folder {STORAGE_DIR_TEST} doesn't exist"
        service = VideoService(STORAGE_DIR_TEST)
        assert isinstance(service, VideoService)
        assert STORAGE_DIR_TEST == service.StorageFolder, f"Storage folder in service is not the same"
        assert os.path.exists(self.videoService.StorageFolder), f"Folder {self.videoService.StorageFolder} doesn't exist"

    def test_ctor_invalid_no_folder(self):
        with self.assertRaises(TypeError):
            VideoService()

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_ctor_invalid_folder_empty(self, empty_folder):
        with self.assertRaises(ValueError):
            VideoService(storage_folder=empty_folder)
    
    def test_ctor_invalid_folder_does_not_exist(self):
        non_existing_folder = "something_random_qsjdkmfjqmsj"
        with self.assertRaises(NotADirectoryError):
            VideoService(storage_folder=non_existing_folder)

    ##################################
    # Test add
    ##################################
    def test_add_valid(self):
        testname = "test_add_valid"
        f = self.folderService.create(testname)
        self.make_video_in_folder(videoname="vid1.mp4", folder=f)
        self.make_video_in_folder(videoname="vid2.mp4", folder=f)
        
        inserted_video = self.videoService.add(name="vid1.mp4", folder=f, frameLength=500)
        inserted_video2 = self.videoService.add(name="vid2.mp4", folder=f, frameLength=500)
        fetched_video = self.videoService.get(id=inserted_video.Id)

        assert isinstance(inserted_video, VideoInfo), f"Video is not an instande of {VideoInfo} got {type(inserted_video)}"
        assert fetched_video == inserted_video, f"Inserted video, is not the same as fetched one"
        assert inserted_video.Name == "vid1.mp4", f"Somewhere hardcoded {inserted_video.Name}"
        assert inserted_video2.Name == "vid2.mp4", f"Somewhere hardcoded {inserted_video2.Name}"
        assert self.videoService.count() == 2, "To much videos seem to be added."

    def test_add_valid_in_nested_folder(self):
        self.make_video_in_folder(videoname="vid.mp4", folder=self.some_nested_folder)

        inserted_video = self.videoService.add(name='vid.mp4', folder=self.some_nested_folder, frameLength=500)

        assert isinstance(inserted_video, VideoInfo), f"Video is not an instande of {VideoInfo} got {type(inserted_video)}"
        assert inserted_video.Name == "vid.mp4", f"Video name is not vid.mp4, got {inserted_video.Name}"

    def test_add_invalid_no_folder(self):
        with self.assertRaises(ValueError):
            self.videoService.add(name="vid.mp4", folder=None, frameLength=500)

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_add_invalid_name_empty(self, empty_name):
        with self.assertRaises(ValueError):
            self.videoService.add(name=empty_name, folder=self.some_folder, frameLength=500)

    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore_extensions())
    def test_create_invalid_only_word_characters_or_numbers(self, invalid_name):
        with self.assertRaises(ValueError):
            self.videoService.add(name=invalid_name, folder=self.some_folder, frameLength=500)
    
    def test_add_in_database_invalid_folderId_does_not_exist_in_database(self):
        testname = "test_add_in_database_invalid_folderId_does_not_exist_in_database"
        self.make_folder_in_storage_dir([testname])
        invalid_folder = Folder(155, testname, None)
        self.make_video_in_folder(videoname="jammer.mp4", folder=invalid_folder)

        with self.assertRaises(LookupError):
            self.videoService.add(name="jammer.mp4", folder=invalid_folder, frameLength=500)

    # Other idea : TODO : check for video orphans in DB
    # For StorageService ?

    ##################################
    # Test exists on drive
    # Params: name & folder
    ##################################
    def test_exists_on_drive_valid_does_exist_in_folder(self):
        inserted_video = self.videoService.add(name="vid.mp4", folder=self.some_folder, frameLength=500)
        assert self.videoService.exists_on_drive(name="vid.mp4", folder=self.some_folder), f"Video vid.mp4 does not exist in {self.some_folder.get_relative_path()}"

    def test_exists_on_drive_valid_does_exist_in_nested_folder(self):
        inserted_video = self.videoService.add(name="vid.mp4", folder=self.some_nested_folder, frameLength=500)
        assert self.videoService.exists_on_drive(name="vid.mp4", folder=self.some_nested_folder), f"Video vid.mp4 does not exist in {self.some_nested_folder.get_relative_path()}"

    def test_exists_on_drive_valid_does_not_exist_in_current_folder_but_has_sibling_which_does_exist(self):
        testname="test_exists_on_drive_valid_does_not_exist_in_current_folder_but_has_sibling_which_does_exist"
        f1 = self.folderService.create(name=testname, parent=None)
        f2 = self.folderService.create(name=f"{testname}_2", parent=None)
        self.make_video_in_folder(videoname="vid_in_siblingfolder.mp4", folder=f2)
        self.videoService.add(name="vid_in_siblingfolder.mp4", folder=f2, frameLength=500)

        assert not self.videoService.exists_on_drive(name="vid_in_siblingfolder.mp4", folder=f1), f"Gave exists because of sibling in {f2.get_relative_path()}"

    def test_exists_on_drive_invalid_does_not_exists(self):
        testname = "test_exists_on_drive_invalid_does_not_exists.mp4"
        assert not self.videoService.exists_on_drive(name=testname, folder=self.some_folder), f"Video {testname} does not exist in {STORAGE_DIR_TEST}"

    def test_exists_on_drive_invalid_does_not_exists_in_nested_folder(self):
        testname = "test_exists_on_drive_invalid_does_not_exists_in_nested_folder.mp4"
        assert not self.videoService.exists_on_drive(name=testname, folder=self.some_nested_folder), f"Video {testname} does not exist in {self.some_nested_folder.get_relative_path()}"

    ##################################
    # Test exists in database (name & folder)
    # Params: id: int = None, name: str = None, folder: Folder = None
    # Default, name is ignored when id is specified
    ##################################
    def test_exists_in_database_valid_name_does_exist(self):
        self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        assert self.videoService.exists_in_database(name=self.vidname, folder=self.some_folder), f"Video {self.vidname} does not exist in database"

    def test_exists_in_database_valid_name_does_exist_in_nested_folder(self):
        self.videoService.add(name=self.vidname, folder=self.some_nested_folder, frameLength=500)
        assert self.videoService.exists_in_database(name=self.vidname, folder=self.some_nested_folder), f"Video {self.vidname} does not exist in database"

    def test_exists_in_database_valid_does_not_exist_in_current_folder_but_has_sibling_which_does_exist(self):
        self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        assert not self.videoService.exists_in_database(name=self.vidname, folder=self.some_nested_folder), f"Video {self.vidname} does not exist in database"

    def test_exists_in_database_invalid_name_does_not_exists(self):
        assert not self.videoService.exists_in_database(name=self.vidname, folder=self.some_folder), f"Video {self.vidname} does exist in database"

    def test_exists_in_database_invalid_name_does_not_exists_in_nested_folder(self):
        assert not self.videoService.exists_in_database(name=self.vidname, folder=self.some_nested_folder), f"Video {self.vidname} does exist in db in {self.some_nested_folder.get_relative_path()}"

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_exists_in_database_invalid_name_empty(self, empty_name):
        with self.assertRaises(ValueError):
            self.videoService.exists_in_database(name=empty_name, folder=self.some_folder)
    
    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore_extensions())
    def test_exists_in_database_invalid_name_empty(self, invalid_name):
        with self.assertRaises(ValueError):
            self.videoService.exists_in_database(name=invalid_name, folder=self.some_folder)

    ##################################
    # Test exists in database (id)
    # Params: id: int = None, name: str = None, folder: Folder = None
    # Default, name is ignored when id is specified
    ##################################
    def test_exists_in_database_valid_id_does_exist(self):
        vidinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        assert self.videoService.exists_in_database(id=vidinfo.Id), f"VideoId {vidinfo.Id} does not exist in database"

    def test_exists_in_database_valid_id_does_exist_in_nested_folder(self):
        vidinfo = self.videoService.add(name=self.vidname, folder=self.some_nested_folder, frameLength=500)
        assert self.videoService.exists_in_database(id=vidinfo.Id), f"VideoId {vidinfo.Id} does not exist in database"

    def test_exists_in_database_invalid_id_does_not_exists(self):
        assert not self.videoService.exists_in_database(id=555), f"FolderId 555 somehow exists in database"

    ##################################
    # Test get (by id)
    ##################################
    def test_get_valid(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        fetched_videoinfo = self.videoService.get(id=inserted_videoinfo.Id)

        assert fetched_videoinfo.Id == inserted_videoinfo.Id, f"Identifiers are not the same"
        assert fetched_videoinfo.Name == inserted_videoinfo.Name, f"Names are not the same"
        assert fetched_videoinfo.Folder == inserted_videoinfo.Folder, f"folders are not the same"

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_get_invalid_id_value(self, invalid_id):
        with self.assertRaises(ValueError):
            self.videoService.get(id=invalid_id)

    def test_get_invalid_id_does_not_exist(self):
        with self.assertRaises(LookupError):
            self.videoService.get(id=155)

    def test_get_valid_loads_frameInfo(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        frameinfo1 = FrameInfo(0, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False)
        frameinfo2 = FrameInfo(250, x=0.4, y=0.5, width=0.8, height=0.6, jumperVisible=True)
        frameinfo3 = FrameInfo(355, x=0.5, y=0.6, width=0.8, height=0.8, jumperVisible=True)
        frameinfo4 = FrameInfo(250, x=0.3, y=0.6, width=0.8, height=0.6, jumperVisible=True)
        frameinfo5 = FrameInfo(255, x=0.3, y=0.6, width=0.8, height=0.6, jumperVisible=True)


        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo1, video=inserted_videoinfo)
        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo2, video=inserted_videoinfo)
        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo3, video=inserted_videoinfo)
        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo4, video=inserted_videoinfo)
        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo5, video=inserted_videoinfo)
        fetched_videoinfo = self.videoService.get(id=inserted_videoinfo.Id)

        assert fetched_videoinfo == inserted_videoinfo, f"Videos differentiated from each other got \n {fetched_videoinfo} \n and \n {inserted_videoinfo}"
        assert len(fetched_videoinfo.Frames) == 4

    # TODO : fetches current skills and labels

    ##################################
    # Test get videos (by folderId)
    ##################################
    def test_get_videos_valid(self):
        testname = "test_get_videos_valid"
        folder = self.folderService.create(name=testname)
        for i in range(5):
            vname = f"child_{i}.mp4"
            self.make_video_in_folder(videoname=vname, folder=folder)
            self.videoService.add(name=vname, folder=folder, frameLength=500)

        vids = self.videoService.get_videos(folderId=folder.Id)

        assert len(vids) == 5, f"folder has not 5 videos"
        numbers = [int(v.Name.split(".")[0].split("_")[1]) for v in vids]
        assert 0 in numbers, f"0 not in numbers"
        assert 1 in numbers, f"1 not in numbers"
        assert 2 in numbers, f"2 not in numbers"
        assert 3 in numbers, f"3 not in numbers"
        assert 4 in numbers, f"4 not in numbers"

    def test_get_videos_valid_no_videos(self):
        testname = "test_get_videos_valid_no_videos"
        folder = self.folderService.create(name=testname)
        videos = self.videoService.get_videos(folderId=folder.Id)

        assert len(videos) == 0, f"Videolist is not empty"

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_get_videos_invalid_id_value(self, invalid_id):
        with self.assertRaises(ValueError):
            self.videoService.get_videos(folderId=invalid_id)

    def test_get_videos_invalid_id_does_not_exist(self):
        with self.assertRaises(LookupError):
            self.videoService.get_videos(folderId=155)


    ##################################
    # Test frameInfo (by frameNr)
    ##################################
    def test_set_frameInfo_valid_new(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        frameinfo = FrameInfo(0, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False)
        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo, video=inserted_videoinfo)
        fetched_videoinfo = self.videoService.get(id=inserted_videoinfo.Id)

        assert fetched_videoinfo == inserted_videoinfo, f"Videos differentiated from each other got \n {fetched_videoinfo} \n and \n {inserted_videoinfo}"
        assert len(fetched_videoinfo.Frames) == 1

    def test_set_frameInfo_valid_update(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        frameinfo2 = FrameInfo(250, x=0.4, y=0.5, width=0.8, height=0.6, jumperVisible=True)
        frameinfo4 = FrameInfo(250, x=0.3, y=0.6, width=0.8, height=0.6, jumperVisible=True)

        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo2, video=inserted_videoinfo)
        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo4, video=inserted_videoinfo)
        fetched_videoinfo = self.videoService.get(id=inserted_videoinfo.Id)

        assert fetched_videoinfo == inserted_videoinfo, f"Videos differentiated from each other got \n {fetched_videoinfo} \n and \n {inserted_videoinfo}"
        assert len(fetched_videoinfo.Frames) == 1

    def test_set_frameInfo_invalid_out_of_bounds(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        frameinfo5 = FrameInfo(555, x=0.3, y=0.6, width=0.8, height=0.6, jumperVisible=True)

        with self.assertRaises(ValueError):
            inserted_videoinfo = self.videoService.set_frameInfo(frameinfo5, video=inserted_videoinfo)

    def test_remove_frameInfo_valid(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        frameinfo = FrameInfo(0, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False)
        inserted_videoinfo = self.videoService.set_frameInfo(frameinfo, video=inserted_videoinfo)
        fetched_videoinfo = self.videoService.get(id=inserted_videoinfo.Id)
        assert len(fetched_videoinfo.Frames) == 1
        assert fetched_videoinfo == inserted_videoinfo, f"Videos differentiated from each other got \n {fetched_videoinfo} \n and \n {inserted_videoinfo}"
        updated_videoInfo = self.videoService.remove_frameInfo(frameNr=frameinfo.FrameNr, video=fetched_videoinfo)
        assert len(updated_videoInfo.Frames) == 0

    def test_remove_frameInfo_invalid_does_not_exists(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        frameinfo = FrameInfo(0, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False)
        assert len(inserted_videoinfo.Frames) == 0
        with self.assertRaises(ValueError):
            updated_videoInfo = self.videoService.remove_frameInfo(frameNr=frameinfo.FrameNr, video=inserted_videoinfo)

    def test_remove_frameInfo_invalid_frameNr_out_of_bounds(self):
        inserted_videoinfo = self.videoService.add(name=self.vidname, folder=self.some_folder, frameLength=500)
        frameinfo = FrameInfo(505, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False)
        
        assert len(inserted_videoinfo.Frames) == 0
        with self.assertRaises(ValueError):
            self.videoService.remove_frameInfo(frameNr=frameinfo.FrameNr, video=inserted_videoinfo)


    ##################################
    # Test rename (by id & new_name)
    # TODO : Should have, implement later
    ##################################
    def test_rename_valid(self):
        # TODO : check that not all folders are renamed
        pass

    def test_rename_invalid_id(self):
        pass

    def test_rename_invalid_name(self):
        pass

    ##################################
    # Test immutable properties
    ##################################
    def test_immutable_property_storage_folder(self):
        with self.assertRaises(AttributeError):
            self.videoService.StorageFolder = "another_folder"
    
    def test_immutable_property_video_repo(self):
        with self.assertRaises(AttributeError):
            self.videoService.VideoRepo = "another_repo"
        
        # Even another object doesn't work
        with self.assertRaises(AttributeError):
            self.videoService.VideoRepo = Folder(1, "folderke")
        
        