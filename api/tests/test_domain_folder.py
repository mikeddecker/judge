# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder
import os

def generate_empty_strings():
    return [ 
        None, "", " ", "\n", "\r", "  ", "\t ", "\t", " \r\n "
    ]
class DomainFolderTestSuite(unittest.TestCase):
    """Domain folder test cases."""
    @parameterized.expand([
        (1, "competition", "main"),
        (2, "competition", "main"),
        (125, "competition", "main"),
        (125, "free", "main"),
        (125, "competition", None),
    ])
    def test_ctor_valid(self, id, name, parentname):
        folderparent = Folder(id, parentname, None) if parentname else None
        folder = Folder(id, name, folderparent)
        self.assertEqual(name, folder.Name, f"Foldername incorrectly initialized {name}, {folder.Name}")
        self.assertEqual(id, folder.Id, f"Folder id incorrectly initialized {id}, {folder.Id}")
        self.assertEqual(folderparent, folder.Parent, f"Parent of folder incorrectly initialized")

    @parameterized.expand(generate_empty_strings())
    def test_ctor_invalid_name_none(self, name):
        with self.assertRaises(ValueError):
            Folder(1, name, None)

    @parameterized.expand([None, 0, -1, -55])
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            Folder(id, "competition", None)

    @parameterized.expand([7, "text", ValueError])
    def test_ctor_invalid_parent_not_a_folder(self, parent):
        with self.assertRaises(ValueError):
            Folder(1, "competition", parent)


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
            folder = Folder(1, "competition", None)
            folder.Parent = Folder(2, "main", None)

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
    # Nice to haves
    ############################################
    def test_ctor_invalid_same_folderId(self):
        pass

if __name__ == '__main__':
    unittest.main()