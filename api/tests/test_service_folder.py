import os

from parameterized import parameterized
from domain.folder import Folder
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

# TODO : test get_root_folders

class FolderServiceTest(TestCase):
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
        pass

    def tearDown(self):
        db.session.remove()
        db.drop_all()
    
    def test_setUp(self):
        assert os.path.exists(self.folderService.StorageFolder), f"Folder {self.folderService.StorageFolder} doesn't exist"
        assert self.folderService.count() == 0, "Databank not empty"

    ##################################
    # Helper functions
    ##################################
    def make_folder_in_storage_dir(self, relative_path : List[str]):
        """
        Joins the relative path, depending on the operating system, then creates the folder.
        """
        # NICE_TO_HAVE : check existence here
        os.mkdir(os.path.join(STORAGE_DIR_TEST, *relative_path))

    ##################################
    # Test constructor
    ##################################
    def test_ctor_valid(self):
        assert os.path.exists(STORAGE_DIR_TEST), f"Folder {STORAGE_DIR_TEST} doesn't exist"
        service = FolderService(STORAGE_DIR_TEST)
        assert isinstance(service, FolderService)
        assert STORAGE_DIR_TEST == service.StorageFolder, f"Storage folder in service is not the same"
        assert os.path.exists(self.folderService.StorageFolder), f"Folder {self.folderService.StorageFolder} doesn't exist"

    def test_ctor_invalid_no_folder(self):
        with self.assertRaises(TypeError):
            FolderService()

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_ctor_invalid_folder_empty(self, empty_folder):
        with self.assertRaises(ValueError):
            FolderService(storage_folder=empty_folder)
    
    def test_ctor_invalid_folder_does_not_exist(self):
        non_existing_folder = "something_random_qsjdkmfjqmsj"
        with self.assertRaises(NotADirectoryError):
            FolderService(storage_folder=non_existing_folder)

    ##################################
    # Test create
    ##################################
    def test_create_valid_without_parent(self):
        testname = "test_create_valid_without_parent"
        created_folder = self.folderService.create(testname, None)

        assert os.path.exists(os.path.join(STORAGE_DIR_TEST, testname)), f"folder {testname} was not created"
        assert created_folder.Id is not None, "Folder id is None"
        assert created_folder.Id != 0, "Folder Id is 0"

        folder_in_db = db.session.query(FolderDB).filter_by(id = created_folder.Id).first()
        assert folder_in_db is not None, "Folder not inserted in DB"
        assert folder_in_db.name == created_folder.Name, "Foldername differs from that in the database"

    def test_create_valid_with_parent(self):
        testname = "test_create_valid_with_parent"
        child = "child"
        parent = self.folderService.create(testname, None)
        created_folder = self.folderService.create(child, parent=parent)

        assert os.path.exists(os.path.join(STORAGE_DIR_TEST, testname, child)), f"folder {child} in {testname} was not created"
        assert os.path.exists(os.path.join(STORAGE_DIR_TEST, created_folder.get_relative_path())), f"folder {created_folder.get_relative_path()} does not exist in {STORAGE_DIR_TEST}"
        assert created_folder.Id is not None, "Folder id is None"
        assert created_folder.Id != 0, "Folder Id is 0"
        assert created_folder.Id != parent.Id, f"Parent id and child id are equal {created_folder.Id}"
        
        folder_in_db = db.session.query(FolderDB).filter_by(id=created_folder.Id).first()
        folder_in_db_parent = folder_in_db.parent # Manually fetch the next one (Because lazy loaded)
        assert folder_in_db is not None, "Folder not inserted in DB"
        assert isinstance(folder_in_db, FolderDB), f"Folder is not {FolderDB}, got {type(folder_in_db)}"
        assert folder_in_db.name == created_folder.Name, "Foldername differs from that in the database"
        assert folder_in_db_parent is not None, f"Parent folder not fetched"
        assert folder_in_db_parent.id == parent.Id, f"ParentIds don't match: {parent.Id} & {folder_in_db.Parent.Id}"

    def test_create_valid_with_nested_parent(self):
        testname = "test_create_valid_with_nested_parent"
        nested1 = "nested1"
        nested2 = "nested2"
        nested3 = "nested3"
        child = "child"
        folder = self.folderService.create(testname, None)
        folder = self.folderService.create(nested1, folder)
        folder = self.folderService.create(nested2, folder)
        folder = self.folderService.create(nested3, folder)
        folder = self.folderService.create(child, folder)
        assert os.path.exists(os.path.join(STORAGE_DIR_TEST, testname, nested1, nested2, nested3, child)), f"folder {child} in {testname} was not created"

    def test_create_valid_has_equal_name_in_other_folder(self):
        testname = "test_create_invalid_has_equal_name_in_other_folder"
        parent1 = "parent1"
        parent2 = "parent2"
        equal_name = "hallelujah"
        folder = self.folderService.create(testname, None)
        p1 = self.folderService.create(parent1, folder)
        p2 = self.folderService.create(parent2, folder)
        self.folderService.create(equal_name, p1)
        self.folderService.create(equal_name, p2)

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_create_invalid_empty_name(self, empty_name):
        with self.assertRaises(ValueError):
            self.folderService.create(empty_name, None)

    def test_create_invalid_already_exists(self):
        testname = "test_create_invalid_already_exists"
        self.folderService.create(testname, None)
        with self.assertRaises(FileExistsError):
            self.folderService.create(testname, None)

    def test_create_invalid_already_exists_with_parent(self):
        testname = "test_create_invalid_already_exists_with_parent"
        p = self.folderService.create(testname, None)
        self.folderService.create("child", parent=p)
        with self.assertRaises(FileExistsError):
            self.folderService.create("child", parent=p)

    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore())
    def test_create_invalid_only_word_characters_or_numbers(self, invalid_name):
        with self.assertRaises(ValueError):
            self.folderService.create(invalid_name, None)
    
    def test_create_invalid_parent_does_not_exist(self):
        testname = "test_create_invalid_parent_does_not_exist"
        # self.folderService.create(testname, None)
        with self.assertRaises(NotADirectoryError):
            self.folderService.create("child", Folder(2, testname, None))

    def test_create_invalid_parent_folder_does_not_exist(self):
        testname = "test_create_invalid_parent_folder_does_not_exist"
        inserted_folder = self.folderService.create(testname, None) # Should get 1, as db is 
        folder = Folder(id=inserted_folder.Id, name="other_name")

        with self.assertRaises(NotADirectoryError):
            self.folderService.create("child", folder)

    def test_create_invalid_name_contains_spaces(self):
        with self.assertRaises(ValueError):
            self.folderService.create("some faulty name", None)

    def test_create_invalid_already_exists(self):
        testname = "test_create_invalid_already_exists"
        self.folderService.create(name=testname)
        # self.folderService.create(testname, None)
        with self.assertRaises(FileExistsError):
            self.folderService.create(name=testname)

    ##################################
    # Test add in database
    ##################################
    def test_add_in_database_valid_without_parent(self):
        testname = "test_add_in_database_valid_without_parent"
        self.make_folder_in_storage_dir([testname])

        inserted_folder = self.folderService.add_in_database(testname, None)
        assert isinstance(inserted_folder, Folder), f"Folder is not an instande of {Folder} got {type(inserted_folder)}"

        folder = Folder(inserted_folder.Id, inserted_folder.Name)
        assert inserted_folder == folder, "Inserted folder does not equal original folder"
        assert self.folderService.count() == 1, "To much folders seem to be added."

    def test_add_in_database_valid_with_parent(self):
        testname = "test_add_in_database_valid_with_parent"
        self.make_folder_in_storage_dir([testname])
        self.make_folder_in_storage_dir([testname, "belgium"])

        parent_folder = self.folderService.add_in_database(testname, None)
        inserted_folder = self.folderService.add_in_database("belgium", parent_folder)
        assert isinstance(inserted_folder, Folder), f"Folder is not an instande of {Folder} got {type(inserted_folder)}"

        folder = Folder(id=inserted_folder.Id, name=inserted_folder.Name, parent=parent_folder)
        assert inserted_folder == folder, "Inserted folder does not equal original folder"
        assert self.folderService.count() == 2, "To much folders seem to be added."

    def test_add_in_database_valid_with_nested_parents(self):
        testname = "test_add_in_database_valid_with_nested_parents"
        self.make_folder_in_storage_dir([testname])
        self.make_folder_in_storage_dir([testname, "belgium"])
        self.make_folder_in_storage_dir([testname, "belgium", "club"])
        self.make_folder_in_storage_dir([testname, "belgium", "club", "training"])
        self.make_folder_in_storage_dir([testname, "belgium", "club", "training", "year"])

        parent_folder = self.folderService.add_in_database(name=testname, parent=None)
        parent_folder = self.folderService.add_in_database(name="belgium", parent=parent_folder)
        parent_folder = self.folderService.add_in_database(name="club", parent=parent_folder)
        parent_folder = self.folderService.add_in_database(name="training", parent=parent_folder)
        inserted_folder = self.folderService.add_in_database("year", parent_folder)

        assert isinstance(inserted_folder, Folder), f"Folder is not an instande of {Folder} got {type(inserted_folder)}"
        folder = Folder(id=inserted_folder.Id, name=inserted_folder.Name, parent=parent_folder)
        assert inserted_folder == folder, "Inserted folder does not equal original folder"
        assert self.folderService.count() == 5, "To much folders seem to be added."

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_add_in_database_invalid_name_empty(self, empty_name):
        with self.assertRaises(ValueError):
            self.folderService.add_in_database(name=empty_name, parent=None)

    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore())
    def test_create_invalid_only_word_characters_or_numbers(self, invalid_name):
        with self.assertRaises(ValueError):
            self.folderService.add_in_database(name=invalid_name, parent=None)
    
    def test_add_in_database_invalid_parent_id_does_not_exist_in_database(self):
        testname = "test_add_in_database_invalid_parent_id_does_not_exist_in_database"
        self.make_folder_in_storage_dir([testname])
        self.make_folder_in_storage_dir([testname, "jammer"])

        parent_folder = Folder(155, testname, None)
        with self.assertRaises(LookupError):
            self.folderService.add_in_database(name="jammer", parent=parent_folder)

    # Other idea : TODO : check for folder orphans in DB
    # For StorageService ?

    ##################################
    # Test exists on drive
    # Params: name & folder
    ##################################
    def test_exists_path_on_drive_valid_does_exist(self):
        testname = "test_exists_valid_does_exist"
        self.make_folder_in_storage_dir([testname])
        assert self.folderService.exists_path_on_drive(name=testname, parent=None), f"Folder {testname} does not exist in {STORAGE_DIR_TEST}"

    def test_exists_path_on_drive_valid_does_exist_with_parent(self):
        testname = "test_exists_path_on_drive_valid_does_exist_with_parent"
        parent_folder = self.folderService.create(testname)
        self.make_folder_in_storage_dir([testname, "child"])
        
        assert self.folderService.exists_path_on_drive(name="child", parent=parent_folder), f"Folder {testname} does not exist in {parent_folder.get_relative_path()}"

    def test_exists_path_on_drive_valid_does_exist_with_nested_parent(self):
        testname = "test_exists_path_on_drive_valid_does_exist_with_nested_parent"
        parent_folder = self.folderService.create(name=testname)
        children = [f"child_{i}" for i in range(10)]
        for c in children:
            parent_folder = self.folderService.create(name=c, parent=parent_folder)   
        self.make_folder_in_storage_dir([testname, *children, "nested"])
        
        assert self.folderService.exists_path_on_drive(name="nested", parent=parent_folder), f"Folder {testname} does not exist in {parent_folder.get_relative_path()}"

    def test_exists_path_on_drive_valid_does_not_exist_in_current_parent_but_has_sibling_which_does_exist(self):
        testname="test_exists_path_on_drive_valid_does_not_exist_in_current_parent_but_has_sibling_which_does_exist"
        p1 = self.folderService.create(name=testname, parent=None)
        p2 = self.folderService.create(name=f"{testname}_2", parent=None)
        self.make_folder_in_storage_dir([f"{testname}_2", "sibling"])

        assert not self.folderService.exists_path_on_drive(name="sibling", parent=p1), f"Gave exists because of sibling in {p2.get_relative_path()}"

    def test_exists_path_on_drive_invalid_does_not_exists(self):
        testname = "test_exists_path_on_drive_invalid_does_not_exists"
        assert not self.folderService.exists_path_on_drive(name=testname, parent=None), f"Folder {testname} does not exist in {STORAGE_DIR_TEST}"

    def test_exists_path_on_drive_invalid_does_not_exists_with_parent(self):
        parent = Folder(2, "something_random_qsdj")
        
        assert not self.folderService.exists_path_on_drive(name="competition", parent=parent), f"Folder does not exist in {parent.get_relative_path()}"

    def test_exists_path_on_drive_invalid_does_not_exists_with_nested_parent(self):
        folders = ["this", "is", "a", "non", "existing", "path"]
        folder = None
        for i, f in enumerate(folders):
            folder = Folder(id=i+1, name=f, parent=folder)
        
        assert not self.folderService.exists_path_on_drive(name="path_path", parent=folder), f"Folder does not exist in {folder.get_relative_path()}"

    ##################################
    # Test exists in database (name)
    # Params: id: int = None, name: str = None, parent: Folder = None
    # Default, name is ignored when id is specified
    ##################################
    def test_exists_in_database_valid_name_does_exist(self):
        testname = "test_exists_in_database_valid_name_does_exist"
        self.folderService.create(name=testname, parent=None)
        assert self.folderService.exists_in_database(name=testname, parent=None), f"Folder {testname} does not exist in database"

    def test_exists_in_database_valid_name_does_exist_with_parent(self):
        testname = "test_exists_in_database_valid_name_does_exist_with_parent"
        parent_folder = self.folderService.create(name=testname, parent=None)
        self.folderService.create(name="child", parent=parent_folder)
        
        assert self.folderService.exists_in_database(name="child", parent=parent_folder), f"Folder {testname} does not exist in database"

    def test_exists_in_database_valid_name_does_exist_with_nested_parent(self):
        testname = "test_exists_in_database_valid_name_does_exist_with_nested_parent"
        parent_folder = self.folderService.create(name=testname, parent=None)
        children = [f"child_{i}" for i in range(10)]
        for c in children:
            parent_folder = self.folderService.create(name=c, parent=parent_folder)   
        self.folderService.create(name="nested", parent=parent_folder)
        
        assert self.folderService.exists_in_database(name="nested", parent=parent_folder), f"Folder {testname} does not exist in {parent_folder.get_relative_path()}"

    def test_exists_in_database_valid_does_not_exist_in_current_parent_but_has_sibling_which_does_exist(self):
        testname="test_exists_in_database_valid_does_not_exist_in_current_parent_but_has_sibling_which_does_exist"
        p1 = self.folderService.create(name=testname, parent=None)
        p2 = self.folderService.create(name=f"{testname}_2", parent=None)
        self.folderService.create(name="sibling", parent=p2)

        assert not self.folderService.exists_in_database(name="sibling", parent=p1), f"Gave exists because of sibling in {p2.get_relative_path()}"

    def test_exists_in_database_invalid_name_does_not_exists(self):
        testname = "test_exists_in_database_invalid_name_does_not_exists"
        assert not self.folderService.exists_in_database(name=testname, parent=None), f"Folder {testname} does not exist in database"

    def test_exists_in_database_invalid_name_does_not_exists_with_parent(self):
        parent_folder = self.folderService.create(name="some_random_name")
        
        assert not self.folderService.exists_in_database(name="some_random_name", parent=parent_folder), f"Folder does not exist in {parent_folder.get_relative_path()}"

    def test_exists_in_database_invalid_name_does_not_exists_with_nested_parent(self):
        folders = ["this", "is", "a", "non", "existing", "path"]
        folder = None
        for i, f in enumerate(folders):
            folder = self.folderService.create(name=f, parent=folder)
        
        assert not self.folderService.exists_in_database(name="path_path", parent=folder), f"Folder does not exist in {folder.get_relative_path()}"

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_exists_in_database_invalid_name_empty(self, empty_folder):
        with self.assertRaises(ValueError):
            self.folderService.exists_in_database(name=empty_folder)
    
    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore())
    def test_exists_in_database_invalid_name_empty(self, invalid_name):
        with self.assertRaises(ValueError):
            self.folderService.exists_in_database(name=invalid_name)

    ##################################
    # Test exists in database (id)
    # Params: id: int = None, name: str = None, parent: Folder = None
    # Default, name is ignored when id is specified
    ##################################
    def test_exists_in_database_valid_id_does_exist(self):
        testname = "test_exists_in_database_valid_id_does_exist"
        assert not self.folderService.exists_in_database(id=1), f"Database not initialized correctly"
        created_folder = self.folderService.create(name=testname, parent=None)

        assert self.folderService.exists_in_database(id=created_folder.Id), f"FolderId {created_folder.Id} does not exist in database"

    def test_exists_in_database_valid_id_does_exist_with_parent(self):
        testname = "test_exists_in_database_valid_id_does_exist_with_parent"
        parent_folder = self.folderService.create(name=testname, parent=None)
        child = self.folderService.create(name="child", parent=parent_folder)
        
        assert self.folderService.exists_in_database(id=child.Id), f"Folder {testname} with id {child.Id} does not exist in database"

    def test_exists_in_database_valid_id_does_exist_with_nested_parent(self):
        testname = "test_exists_in_database_valid_id_does_exist_with_nested_parent"
        parent_folder = self.folderService.create(name=testname, parent=None)
        children = [f"child_{i}" for i in range(10)]
        for c in children:
            parent_folder = self.folderService.create(name=c, parent=parent_folder)   
        created_folder = self.folderService.create(name="nested", parent=parent_folder)
        
        assert self.folderService.exists_in_database(id=created_folder.Id), f"FolderId {created_folder.Id} does not exist in database"

    def test_exists_in_database_invalid_id_does_not_exists(self):
        assert not self.folderService.exists_in_database(id=555), f"FolderId 555 somehow exists in database"

    ##################################
    # Test get (by id)
    # Only by id + combo get_children
    ##################################

    def test_get_valid(self):
        testname = "test_get_valid"
        created_folder = self.folderService.create(name=testname)
        fetched_created_folder = self.folderService.get(id=created_folder.Id)

        assert fetched_created_folder.Id == created_folder.Id, f"Identifiers are not the same"
        assert fetched_created_folder.Name == created_folder.Name, f"Names are not the same"
        assert fetched_created_folder.Parent == created_folder.Parent, f"Parents are not the same"

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_get_invalid_id_value(self, invalid_id):
        with self.assertRaises(ValueError):
            self.folderService.get(id=invalid_id)

    def test_get_invalid_id_does_not_exist(self):
        with self.assertRaises(LookupError):
            self.folderService.get(id=155)

    ##################################
    # Test get children (by id)
    # Only by id + combo get_children
    ##################################
    def test_get_children_valid(self):
        testname = "test_get_children_valid"
        parent = self.folderService.create(name=testname)
        for i in range(5):
            self.folderService.create(name=f"child_{i}", parent=parent)

        children = self.folderService.get_children(id=parent.Id)

        assert len(children) == 5, f"Parent has not 5 children"
        numbers = [int((c.Name).split("_")[1]) for c in children]
        assert 0 in numbers, f"0 not in numbers"
        assert 1 in numbers, f"1 not in numbers"
        assert 2 in numbers, f"2 not in numbers"
        assert 3 in numbers, f"3 not in numbers"
        assert 4 in numbers, f"4 not in numbers"

    def test_get_children_valid_no_children(self):
        testname = "test_get_children_valid_no_children"
        parent = self.folderService.create(name=testname)
        children = self.folderService.get_children(id=parent.Id)

        assert len(children) == 0, f"Children is not empty"

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_get_children_invalid_id_value(self, invalid_id):
        with self.assertRaises(ValueError):
            self.folderService.get_children(id=invalid_id)

    def test_get_children_invalid_id_does_not_exist(self):
        with self.assertRaises(LookupError):
            self.folderService.get_children(id=155)


    ##################################
    # Test delete (by id)
    ##################################
    def test_delete_valid(self):
        testname = "test_delete_valid"
        f = self.folderService.create(name=testname)

        assert self.folderService.exists_in_database(id=f.Id), f"Folder {f.Id} doesn't exist in the database"
        assert self.folderService.exists_in_database(name=testname), f"Folder {testname} does not exist in the database"
        assert self.folderService.exists_path_on_drive(name=testname), f"Folder {testname} does not exist in {STORAGE_DIR_TEST}"

        self.folderService.delete(id=f.Id)

        assert not self.folderService.exists_in_database(id=f.Id), f"Folder {f.Id} still exists in the database"
        assert not self.folderService.exists_in_database(name=testname), f"Folder {testname} still exists in the database"
        assert not self.folderService.exists_path_on_drive(name=testname), f"Folder {testname} still exists in {STORAGE_DIR_TEST}"


    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_delete_invalid_id_value(self, invalid_id):
        with self.assertRaises(ValueError):
            self.folderService.delete(id=invalid_id)

    def test_delete_invalid_id_does_not_exist(self):
        with self.assertRaises(LookupError):
            self.folderService.delete(id=155)

    def test_delete_invalid_has_children(self):
        testname = "test_delete_invalid_has_children"
        p = self.folderService.create(name=testname)
        f = self.folderService.create(name="child", parent=p)

        with self.assertRaises(PermissionError):
            self.folderService.delete(id=p.Id)

    def test_delete_invalid_has_videos(self):
        testname = "test_delete_invalid_has_videos"
        p = self.folderService.create(name=testname)
        videoname = 'empty_video.mp4'
        with open(os.path.join(STORAGE_DIR_TEST, testname, videoname), 'w') as fp:
            pass
        self.videoService.add(name=videoname, folder=p, frameLength=500)

        with self.assertRaises(PermissionError):
            self.folderService.delete(id=p.Id)

    def test_delete_invalid_has_other_content(self):
        testname = "test_delete_invalid_has_other_content"
        p = self.folderService.create(name=testname)
        with open(os.path.join(STORAGE_DIR_TEST, testname, 'undiscovered_video.mp4'), 'w') as fp:
            pass
        with open(os.path.join(STORAGE_DIR_TEST, testname, 'textfile'), 'w') as fp:
            pass

        with self.assertRaises(PermissionError):
            self.folderService.delete(id=p.Id)

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
            self.folderService.StorageFolder = "another_folder"
    
    def test_immutable_property_folder_repo(self):
        with self.assertRaises(AttributeError):
            self.folderService.FolderRepo = "another_repo"
        
        # Even another object doesn't work
        with self.assertRaises(AttributeError):
            self.folderService.FolderRepo = Folder(1, "folderke")