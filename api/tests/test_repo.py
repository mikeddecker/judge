from domain.folder import Folder
from dotenv import load_dotenv
from flask import Flask, current_app
from flask_migrate import Migrate
from flask_testing import TestCase
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.models import Folder as FolderDB
from repository.MapToDomain import MapToDomain

load_dotenv()

# TODO : update tests, just a tryout to make it work
class MyTest(TestCase):
    def create_app(self):
        app = Flask(__name__)
        app.config.from_object("config.TestConfig")
        app.json.compact = False
        db.init_app(app)
        migrate = Migrate(app, current_app)
        migrate.init_app(app, db)
        with app.app_context():
            db.create_all()
        
        self.repo = FolderRepository(db=db)
        return app
    
    def test_add_valid_without_parent(self):
        name="competition"

        # Pre-check
        count_competitionFolders = db.session.query(FolderDB).filter(FolderDB.name == name).count()
        assert count_competitionFolders == 0, "Databank not empty"

        # Act
        self.repo.add("competition", None)
        competitionFolders = db.session.query(FolderDB).filter(FolderDB.name == name).all()
        
        assert len(competitionFolders) == 1, "Folder competition not inserted"
        inserted_folderDB = competitionFolders[0]
        inserted_folder = MapToDomain.map_folder(inserted_folderDB)
        assert isinstance(inserted_folder, Folder), f"Folder is not an instande of {Folder} got {type(inserted_folder)}"
        folder = Folder(inserted_folder.Id, inserted_folder.Name)
        assert inserted_folder == folder, "Inserted folder does not equal original folder"

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

    def test_server_count(self):
        for i in range(5):
            folder = FolderDB(name=f"competition{i}")
            db.session.add(folder)
        assert 5 == db.session.query(FolderDB.id).count()

    def test_server_count2(self):
        for i in range(5):
            folder = FolderDB(name=f"competition{i + 5}")
            db.session.add(folder)
        assert 5 == db.session.query(FolderDB.id).count()

    def test_contains_competition(self):
        K = 3
        folder = FolderDB(name=f"comp_{1}")
        db.session.add(folder)
        for i in range(K):
            folder = FolderDB(name=f"random_name_{i}", parentId=i+1)
            db.session.add(folder)
        for i in range(4):
            folder = FolderDB(name=f"other_name_{i}")
            db.session.add(folder)

        ls = db.session.query(FolderDB).filter(FolderDB.name == "random_name_2").all()
        print("LS", ls)
        assert 1 == len(ls)


    def setUp(self):
        print("setup calles.")
        pass

    def tearDown(self):
        db.session.remove()
        db.drop_all()