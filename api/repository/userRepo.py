from repository.db import db
from repository.models import User as UserDB
from domain.user import User as UserDomain
from datetime import datetime
from MapToDomain import MapToDomain

class UserRepo:
    @staticmethod
    def create_user(email: str, firstName: str, lastName: str, passwordHash: str, salt: str) -> UserDomain:
        """Create a new user"""
        user = UserDB(
            email=email,
            firstName=firstName,
            lastName=lastName,
            passwordHash=passwordHash,
            salt=salt,
            createdAt=datetime.now(),
            updatedAt=datetime.now()
        )
        db.session.add(user)
        db.session.commit()
        return MapToDomain.map_user(user)

    @staticmethod
    def get_user_by_email(email: str) -> UserDomain:
        """Get user by email"""
        user = UserDB.query.filter_by(email=email).first()
        if not user:
            return None
        return MapToDomain.map_user(user)

    @staticmethod
    def get_user_by_id(user_id: int) -> UserDomain:
        """Get user by ID"""
        user = UserDB.query.get(user_id)
        if not user:
            return None
        return MapToDomain.map_user(user)

    @staticmethod
    def update_lastLogin(user_id: int) -> None:
        """Update user's last login timestamp"""
        user = UserDB.query.get(user_id)
        if user:
            user.lastLogin = datetime.now()
            user.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def update_password(user_id: int, passwordHash: str, salt: str) -> None:
        """Update user's password"""
        user = UserDB.query.get(user_id)
        if user:
            user.passwordHash = passwordHash
            user.salt = salt
            user.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def set_mfaCode(user_id: int, mfaCode: str, expires_at: datetime) -> None:
        """Set MFA code for user"""
        user = UserDB.query.get(user_id)
        if user:
            user.mfaCode = mfaCode
            user.mfaCodeExpires = expires_at
            db.session.commit()

    @staticmethod
    def enable_mfa(user_id: int) -> None:
        """Enable MFA for user"""
        user = UserDB.query.get(user_id)
        if user:
            user.mfaEnabled = True
            user.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def verify_mfaCode(user_id: int, mfaCode: str) -> bool:
        """Verify MFA code"""
        user = UserDB.query.get(user_id)
        if not user or not user.mfaCode:
            return False
        if user.mfaCodeExpires and datetime.now() > user.mfaCodeExpires:
            return False
        return user.mfaCode == mfaCode

    @staticmethod
    def user_exists(email: str) -> bool:
        """Check if user with email exists"""
        return UserDB.query.filter_by(email=email).first() is not None

    @staticmethod
    def get_all_users():
        """Get all users (admin only)"""
        users = UserDB.query.all()
        return [ MapToDomain.map_user(user) for user in users]

