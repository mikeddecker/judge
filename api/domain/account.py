from datetime import datetime

class Account:
    def __init__(
            self,
            email: str,
            firstName: str,
            lastName: str,
            passwordHash: str,
            salt: str,
            id: int = None,
            lastLogin: datetime = None,
            createdAt: datetime = None,
            updatedAt: datetime = None,
            mfaEnabled: bool = False,
            mfaCode: str = None,
    ) -> None:
        """
        Account domain model

        :param email: Account email address (unique)
        :param firstName: Account first name
        :param lastName: Account last name
        :param passwordHash: Hashed password
        :param salt: Salt used for password hashing
        :param id: Account database identifier
        :param lastLogin: Last login timestamp
        :param createdAt: Account creation timestamp
        :param updatedAt: Last update timestamp
        :param mfaEnabled: Whether MFA is enabled
        :param mfaCode: Current MFA code
        """
        self.id = id
        self.email = email
        self.firstName = firstName
        self.lastName = lastName
        self.passwordHash = passwordHash
        self.salt = salt
        self.lastLogin = lastLogin
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.mfaEnabled = mfaEnabled
        self.mfaCode = mfaCode

    def to_dict(self):
        return {
            'id': self.id.hex(),
            'email': self.email,
            'firstName': self.firstName,
            'lastName': self.lastName,
            'lastLogin': self.lastLogin.isoformat() if self.lastLogin else None,
            'createdAt': self.createdAt.isoformat() if self.createdAt else None,
            'updatedAt': self.updatedAt.isoformat() if self.updatedAt else None,
            'mfaEnabled': self.mfaEnabled,
        }

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())
