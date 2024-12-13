import os
import unittest

from parameterized import parameterized
from domain.folder import Folder
from dotenv import load_dotenv
from flask import Flask, current_app
from flask_migrate import Migrate
from flask_testing import TestCase
from repository.db import db
from services.folderService import FolderService
from tests.TestHelper import TestHelper
from typing import List

load_dotenv()
STORAGE_DIR_TEST = os.getenv("STORAGE_DIR_TEST") 
if os.path.exists(STORAGE_DIR_TEST):
    os.system(f"rm -rf {STORAGE_DIR_TEST}/*")
else:
    os.mkdir(STORAGE_DIR_TEST)

# TODO : update tests, just a tryout to make it work
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
        return app
    
    def setUp(self):
        pass

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        
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
        pass

    ##################################
    # Test create
    ##################################
    def test_create_on_drive_valid_without_parent(self):
        assert self.folderService.count() == 0, "Databank not empty"

        testname = "test_create_on_drive_valid_without_parent"
        self.folderService.create_on_drive(testname, None)

        assert os.path.exists(os.path.join(STORAGE_DIR_TEST, testname)), f"folder {testname} was not created"

    def test_create_on_drive_valid_with_parent(self):
        assert self.folderService.count() == 0, "Databank not empty"

        testname = "test_create_on_drive_valid_with_parent"
        child = "child"
        self.folderService.create_on_drive(testname, None)
        self.folderService.create_on_drive(child, Folder(1, testname))
        assert os.path.exists(os.path.join(STORAGE_DIR_TEST, testname, child)), f"folder {child} in {testname} was not created"

    def test_create_on_drive_valid_with_nested_parent(self):
        assert self.folderService.count() == 0, "Databank not empty"

        testname = "test_create_on_drive_valid_with_nested_parent"
        nested1 = "nested1"
        nested2 = "nested2"
        nested3 = "nested3"
        child = "child"
        folder = self.folderService.create_on_drive(testname, None)
        folder = self.folderService.create_on_drive(nested1, folder)
        folder = self.folderService.create_on_drive(nested2, folder)
        folder = self.folderService.create_on_drive(nested3, folder)
        folder = self.folderService.create_on_drive(child, folder)
        print(folder.get_relative_path())
        assert os.path.exists(os.path.join(STORAGE_DIR_TEST, testname, nested1, nested2, nested3, child)), f"folder {child} in {testname} was not created"

    def test_create_on_drive_valid_has_equal_name_in_other_folder(self):
        assert self.folderService.count() == 0, "Databank not empty"

        testname = "test_create_on_drive_invalid_has_equal_name_in_other_folder"
        parent1 = "parent1"
        parent2 = "parent2"
        equal_name = "hallelujah"
        folder = self.folderService.create_on_drive(testname, None)
        p1 = self.folderService.create_on_drive(parent1, folder)
        p2 = self.folderService.create_on_drive(parent2, folder)
        self.folderService.create_on_drive(equal_name, p1)
        self.folderService.create_on_drive(equal_name, p2)

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_create_on_drive_invalid_empty_name(self, empty_name):
        assert self.folderService.count() == 0, "Databank not empty"

        with self.assertRaises(ValueError):
            self.folderService.create_on_drive(empty_name, None)

    def test_create_on_drive_invalid_already_exists(self):
        assert self.folderService.count() == 0, "Databank not empty"

        testname = "test_create_on_drive_invalid_already_exists"
        self.folderService.create_on_drive(testname, None)
        with self.assertRaises(FileExistsError):
            self.folderService.create_on_drive(testname, None)

    @parameterized.expand(["hello!", "dotted.name", "seme%", "0623()", "§dsqk"])
    def test_create_on_drive_invalid_only_word_characters_or_numbers(self, invalid_name):
        assert self.folderService.count() == 0, "Databank not empty"

        with self.assertRaises(ValueError):
            self.folderService.create_on_drive(invalid_name, None)
    
    def test_create_on_drive_invalid_parent_does_not_exist(self):
        assert self.folderService.count() == 0, "Databank not empty"

        testname = "test_create_on_drive_invalid_parent_does_not_exist"
        # self.folderService.create_on_drive(testname, None)
        with self.assertRaises(NotADirectoryError):
            self.folderService.create_on_drive("child", Folder(2, testname, None))

    # TODO : add parts that it also adds in in the database.
    # TODO : add parts that add to database, checks existence of folder
    # Extra : just in case
    def test_create_on_drive_invalid_parent_id_and_foldername_in_database_does_not_match(self):
        assert self.folderService.count() == 0, "Databank not empty"

        # testname = "test_create_on_drive_invalid_parent_id_and_foldername_in_database_does_not_match"
        # inserted_folder = self.folderService.create_on_drive(testname, None) # Should get 1, as db is 
        # folder = Folder(id=inserted_folder.Id + 5, name=inserted_folder.Name)

        # with self.assertRaises(LookupError):
        #     self.folderService.create_on_drive("child", folder)
        pass

    ##################################
    # Test add
    ##################################
    def test_add_in_database_valid_without_parent(self):
        # Pre-check
        assert self.folderService.count() == 0, "Databank not empty"

        self.make_folder_in_storage_dir(["competition"])

        inserted_folder = self.folderService.add_in_database("competition", None)
        assert isinstance(inserted_folder, Folder), f"Folder is not an instande of {Folder} got {type(inserted_folder)}"

        folder = Folder(inserted_folder.Id, inserted_folder.Name)
        assert inserted_folder == folder, "Inserted folder does not equal original folder"
        assert self.folderService.count() == 1, "To much folders seem to be added."

    def test_add_valid_with_parent(self):
        pass

    def test_add_valid_with_nested_parents(self):
        pass

    def test_add_invalid_name_empty(self):
        pass

    def test_add_invalid_parent_without_id(self):
        pass

    def test_add_invalid_parent_invalid_id(self):
        pass

    def test_add_invalid_parent_id_does_not_exist(self):
        pass

    def test_add_invalid_nested_parent_invalid(self):
        # let's skip this
        # As long as the parent exists (which will be valid)
        # and the folder effectively exists (checked by the FolderService)
        # Then it can be fetched
        # Other idea : TODO : check for folder orphans in DB
        pass

    def test_exists_valid_does_exist(self):
        pass

    def test_exists_valid_does_not_exists(self):
        pass

    def test_exists_invalid__id(self):
        pass

    def test_get_valid(self):
        pass

    def test_get_valid_does_not_exist_nonetype(self):
        pass

    def test_get_invalid__id(self):
        pass

    def test_exists_by_name(self):
        # When needed
        pass

    def test_get_by_name(self):
        # When needed
        pass

    def test_delete_valid(self):
        pass

    def test_delete_invalid__id(self):
        pass

    def test_delete_invalid_has_children(self):
        pass

    def test_rename_valid(self):
        # TODO : check that not all folders are renamed
        pass

    def test_rename_invalid_id(self):
        pass

    def test_rename_invalid_name(self):
        pass
