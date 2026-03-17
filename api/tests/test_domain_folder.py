# -*- coding: utf-8 -*-
import os
import unittest
from uuid import UUID, uuid4

from domain.folder import Folder
from parameterized import parameterized
from tests.TestHelper import TestHelper

# Fixed UUID pairs for deterministic parameterized tests
_F1A = UUID('00000000-0000-0000-0000-000000000101')
_F1B = UUID('00000000-0000-0000-0000-000000000102')
_F2A = UUID('00000000-0000-0000-0000-000000000201')
_F2B = UUID('00000000-0000-0000-0000-000000000202')
_F3A = UUID('00000000-0000-0000-0000-000000000301')
_F3B = UUID('00000000-0000-0000-0000-000000000302')
_F4A = UUID('00000000-0000-0000-0000-000000000401')
_F4B = UUID('00000000-0000-0000-0000-000000000402')
_F5B = UUID('00000000-0000-0000-0000-000000000502')

class DomainFolderTestSuite(unittest.TestCase):
    """Domain folder test cases."""
    @parameterized.expand([
        (_F1A, _F1B, "competition", "main"),
        (_F2A, _F2B, "competition", "main"),
        (_F3A, _F3B, "competition", "main"),
        (_F4A, _F4B, "free", "main"),
        (_F5B, None, "competition", None),
    ])
    def test_ctor_valid(self, parent_id, child_id, name, parentname):
        folderparent = Folder(parent_id, parentname, None) if parentname else None
        folder_id = child_id if child_id else parent_id
        folder = Folder(folder_id, name, folderparent)
        self.assertEqual(name, folder.Name, f"Foldername incorrectly initialized {name}, {folder.Name}")
        self.assertEqual(folder_id, folder.Id, f"Folder id incorrectly initialized {folder_id}, {folder.Id}")
        self.assertEqual(folderparent, folder.Parent, f"Parent of folder incorrectly initialized")

    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_ctor_invalid_name_none(self, name):
        with self.assertRaises(ValueError):
            Folder(1, name, None)

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            Folder(id, "competition", None)

    @parameterized.expand([7, "text", ValueError])
    def test_ctor_invalid_parent_not_a_folder(self, parent):
        with self.assertRaises(ValueError):
            Folder(1, "competition", parent)

    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore())
    def test_ctor_invalid_only_word_characters_or_numbers(self, invalid_name):
        with self.assertRaises(ValueError):
            Folder(id=1, name=invalid_name, parent=None)

    ############################################
    # Properties
    ############################################
    def test_change_id_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(uuid4(), "competition", None)
            folder.Id = uuid4()

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(uuid4(), "competition", None)
            folder.Name = "free"

    def test_change_parent_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(uuid4(), "competition", Folder(uuid4(), "not_none", None))
            folder.Parent = Folder(uuid4(), "main", None)
    
    def test_change_parent_immutable_except_none_is_changeable(self):
        parent_folder = Folder(uuid4(), "main", None)
        folder = Folder(uuid4(), "competition", None)
        folder.Parent = parent_folder
        self.assertEqual(folder.Parent, parent_folder)

    ############################################
    # Methods
    ############################################
    def test_get_relative_path_no_parent(self):
        folder = Folder(uuid4(), "competition", None)
        self.assertEqual("competition", folder.get_relative_path())
    
    def test_get_relative_path_one_parent(self):
        main = Folder(uuid4(), "main", None)
        comp = Folder(uuid4(), "competition", main)
        path = os.path.join("main", "competition")
        self.assertEqual(path, comp.get_relative_path())
    
    def test_get_relative_path_multi_parent(self):
        folder = Folder(uuid4(), "root", None)
        foldernames = ["root"]
        for i in range(2, 12):
            folder = Folder(uuid4(), f"subfolder{i}", folder)
            foldernames.append(f"subfolder{i}")
        path = os.path.join(*foldernames)
        self.assertEqual(path, folder.get_relative_path())

    ############################################
    # General Methods
    ############################################
    @parameterized.expand([
        (UUID('00000000-0000-0000-0001-000000000001'), UUID('00000000-0000-0000-0001-000000000001'), "competition", "competition"),
        (UUID('00000000-0000-0000-0002-000000000002'), UUID('00000000-0000-0000-0002-000000000002'), "competition", "competition"),
        (UUID('00000000-0000-0000-0001-000000000001'), UUID('00000000-0000-0000-0001-000000000001'), "belgium", "belgium"),
    ])
    def test_equals_valid_name(self, id_1, id_2, name_1, name_2):
        folder1 = Folder(id=id_1, name=name_1)
        folder2 = Folder(id=id_2, name=name_2)
        self.assertEqual(folder1, folder2)

    def test_equals_valid_with_parent(self):
        shared_parent_id = UUID('00000000-0000-0000-0003-000000000003')
        shared_child_id = UUID('00000000-0000-0000-0001-000000000001')
        parent_1 = Folder(id=shared_parent_id, name="parentfolder_hahaha")
        parent_2 = Folder(id=shared_parent_id, name="parentfolder_hahaha")
        folder1 = Folder(id=shared_child_id, name="name_1", parent=parent_1)
        folder2 = Folder(id=shared_child_id, name="name_1", parent=parent_2)
        self.assertEqual(folder1, folder2)

    def test_equals_valid_nested_parent(self):
        pass

    def test_equals_invalid_name_and_id(self):
        folder1 = Folder(id=UUID('00000000-0000-0000-0001-000000000001'), name="competition")
        folder2 = Folder(id=UUID('00000000-0000-0000-0002-000000000002'), name="competition")
        self.assertNotEqual(folder1, folder2)

    def test_equals_invalid_with_parent(self):
        parent_1 = Folder(id=UUID('00000000-0000-0000-0003-000000000003'), name="parent1")
        parent_2 = Folder(id=UUID('00000000-0000-0000-0004-000000000004'), name="parent2")
        folder1 = Folder(id=UUID('00000000-0000-0000-0001-000000000001'), name="name_1", parent=parent_1)
        folder2 = Folder(id=UUID('00000000-0000-0000-0001-000000000001'), name="name_1", parent=parent_2)
        self.assertNotEqual(folder1, folder2)

    def test_equals_invalid_nested_parent(self):
        grandparent1 = Folder(id=UUID('00000000-0000-0000-0005-000000000005'), name="grand1")
        grandparent2 = Folder(id=UUID('00000000-0000-0000-0006-000000000006'), name="grand2")
        parent1 = Folder(id=UUID('00000000-0000-0000-0003-000000000003'), name="parent", parent=grandparent1)
        parent2 = Folder(id=UUID('00000000-0000-0000-0004-000000000004'), name="parent", parent=grandparent2)
        folder1 = Folder(id=UUID('00000000-0000-0000-0001-000000000001'), name="child", parent=parent1)
        folder2 = Folder(id=UUID('00000000-0000-0000-0001-000000000001'), name="child", parent=parent2)
        self.assertNotEqual(folder1, folder2)

    ############################################
    # Nice to haves
    ############################################
    def test_ctor_invalid_parent_has_same_folderId(self):
        shared_id = UUID('00000000-0000-0000-0001-000000000001')
        parent = Folder(id=shared_id, name="parent")
        with self.assertRaises(ValueError):
            Folder(id=shared_id, name="child", parent=parent)

    def test_ctor_invalid_nested_parent_has_same_folderId(self):
        shared_id = UUID('00000000-0000-0000-0001-000000000001')
        grandparent = Folder(id=shared_id, name="grandparent")
        parent = Folder(id=UUID('00000000-0000-0000-0002-000000000002'), name="parent", parent=grandparent)
        with self.assertRaises(ValueError):
            Folder(id=shared_id, name="child", parent=parent)

if __name__ == '__main__':
    unittest.main()

