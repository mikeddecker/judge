# -*- coding: utf-8 -*-
import os
import unittest

from domain.folder import Folder
from parameterized import parameterized
from tests.TestHelper import TestHelper

class DomainFolderTestSuite(unittest.TestCase):
    """Domain folder test cases."""
    @parameterized.expand([
        (1, "competition", "main"),
        (2, "competition", "main"),
        (125_000, "competition", "main"),
        (125_000, "free", "main"),
        (125_000, "competition", None),
    ])
    def test_ctor_valid(self, id, name, parentname):
        folderparent = Folder(id, parentname, None) if parentname else None
        folder = Folder(id, name, folderparent)
        self.assertEqual(name, folder.Name, f"Foldername incorrectly initialized {name}, {folder.Name}")
        self.assertEqual(id, folder.Id, f"Folder id incorrectly initialized {id}, {folder.Id}")
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
            folder = Folder(1, "competition", None)
            folder.Id = 7

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(1, "competition", None)
            folder.Name = "free"

    def test_change_parent_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(1, "competition", Folder(3, "not_none", None))
            folder.Parent = Folder(2, "main", None)
    
    def test_change_parent_immutable_except_none_is_changeable(self):
        parent_folder = Folder(2, "main", None)
        folder = Folder(1, "competition", None)
        folder.Parent = parent_folder
        self.assertEqual(folder.Parent, parent_folder)

    ############################################
    # Methods
    ############################################
    def test_get_relative_path_no_parent(self):
        folder = Folder(1, "competition", None)
        self.assertEqual("competition", folder.get_relative_path())
    
    def test_get_relative_path_one_parent(self):
        main = Folder(1, "main", None)
        comp = Folder(1, "competition", main)
        path = os.path.join("main", "competition")
        self.assertEqual("main/competition", comp.get_relative_path())
    
    def test_get_relative_path_multi_parent(self):
        folder = Folder(1, "root", None)
        foldernames = ["root"]
        for i in range(10):
            j = i + 1
            folder = Folder(j, f"subfolder{j}", folder)
            foldernames.append(f"subfolder{j}")
        path = os.path.join(*foldernames)
        self.assertEqual(path, folder.get_relative_path())

    ############################################
    # General Methods
    ############################################
    @parameterized.expand([
        (1, 1, "competition", "competition"),
        (2, 2, "competition", "competition"),
        (1, 1, "belgium", "belgium"),
    ])
    def test_equals_valid_name(self, id_1, id_2, name_1, name_2):
        folder1 = Folder(id=id_1, name=name_1)
        folder2 = Folder(id=id_2, name=name_2)
        self.assertEqual(folder1, folder2)

    def test_equals_valid_with_parent(self):
        parent_1 = Folder(id=3, name="parentfolder_hahaha")
        parent_2 = Folder(id=3, name="parentfolder_hahaha")
        folder1 = Folder(id=1, name="name_1", parent=parent_1)
        folder2 = Folder(id=1, name="name_1", parent=parent_2)
        self.assertEqual(folder1, folder2)

    def test_equals_valid_nested_parent(self):
        pass

    def test_equals_invalid_name_and_id(self):
        pass

    def test_equals_invalid_with_parent(self):
        pass

    def test_equals_invalid_nested_parent(self):
        pass

    ############################################
    # Nice to haves
    ############################################
    def test_ctor_invalid_parent_has_same_folderId(self):
        pass

    def test_ctor_invalid_nested_parent_has_same_folderId(self):
        pass

if __name__ == '__main__':
    unittest.main()