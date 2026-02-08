from repository.db import db
from repository.models import User
from domain.user import User as UserDomain
from datetime import datetime

class UserRepo:
    @staticmethod
    def create_user(email: str, firstName: str, lastName: str, passwordHash: str, salt: str) -> UserDomain:
        """Create a new user"""
        user = User(
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
        return UserRepo._map_to_domain(user)

    @staticmethod
    def get_user_by_email(email: str) -> UserDomain:
        """Get user by email"""
        user = User.query.filter_by(email=email).first()
        if not user:
            return None
        return UserRepo._map_to_domain(user)

    @staticmethod
    def get_user_by_id(user_id: int) -> UserDomain:
        """Get user by ID"""
        user = User.query.get(user_id)
        if not user:
            return None
        return UserRepo._map_to_domain(user)

    @staticmethod
    def update_lastLogin(user_id: int) -> None:
        """Update user's last login timestamp"""
        user = User.query.get(user_id)
        if user:
            user.lastLogin = datetime.now()
            user.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def update_password(user_id: int, passwordHash: str, salt: str) -> None:
        """Update user's password"""
        user = User.query.get(user_id)
        if user:
            user.passwordHash = passwordHash
            user.salt = salt
            user.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def set_mfaCode(user_id: int, mfaCode: str, expires_at: datetime) -> None:
        """Set MFA code for user"""
        user = User.query.get(user_id)
        if user:
            user.mfaCode = mfaCode
            user.mfaCodeExpires = expires_at
            db.session.commit()

    @staticmethod
    def enable_mfa(user_id: int) -> None:
        """Enable MFA for user"""
        user = User.query.get(user_id)
        if user:
            user.mfaEnabled = True
            user.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def verify_mfaCode(user_id: int, mfaCode: str) -> bool:
        """Verify MFA code"""
        user = User.query.get(user_id)
        if not user or not user.mfaCode:
            return False
        if user.mfaCodeExpires and datetime.now() > user.mfaCodeExpires:
            return False
        return user.mfaCode == mfaCode

    @staticmethod
    def user_exists(email: str) -> bool:
        """Check if user with email exists"""
        return User.query.filter_by(email=email).first() is not None

    @staticmethod
    def get_all_users():
        """Get all users (admin only)"""
        users = User.query.all()
        return [UserRepo._map_to_domain(user) for user in users]

    @staticmethod
    def _map_to_domain(user: User) -> UserDomain:
        """Map database model to domain model"""
        return UserDomain(
            id=user.id,
            email=user.email,
            firstName=user.firstName,
            lastName=user.lastName,
            passwordHash=user.passwordHash,
            salt=user.salt,
            lastLogin=user.lastLogin,
            createdAt=user.createdAt,
            updatedAt=user.updatedAt,
            mfaEnabled=user.mfaEnabled,
            mfaCode=user.mfaCode,
        )

