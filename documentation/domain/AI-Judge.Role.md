The roles an account can have. 
Roles will al be listed, some may or may not have an effect, because of overruling.
By default, roles are least privilege -> absent.

# Roles

- 📖/✍ Access to **videos** of another account (CRUD)
	- Upload
	- Read
		- By friends
		- By organisation
	- Update
	- Delete
- 📖/✍ Access to **train launch page**
	- Train status
	- Can train models (💲)
- 📖/✍ Access to video labels
	- Create
		- By friends
		- By organisation (representative)
		- By organisation (member/...)?
	- Read
		- By friends
		- By organisation (representative)
		- By organisation (member/...)
		- By jumper in the video 🔽
	- Update (request)
		- By friends
		- By organisation (representative)
		- By organisation (member/...)
	- Delete (request)
		- By friends
		- By organisation (representative)
		- By organisation (member/...)
- 📖/✍ Can define Layers
	- Inherit?
- 📖/✍ Can define Layers combinations
	- Inherit?
- 📖/✍ Can define Tags
	- Inherit?
- 📖/✍ ...

Idea: Roles on specific folders
-> Which is more specific

# Account - Properties:

- Uuid
- RoleGiver (nullable)
- RoleTaker
- GivenRole
- permissionGranted (default yes)
- CreatedAt
- UpdatedAt
- AccountUuid (in case of specific friend/)

⚠  An admin can see everything (because of switchable to other accounts).

