from repository.db import db
from repository.models import Account as AccountDB
from domain.account import Account as AccountDomain
from datetime import datetime
from repository.MapToDomain import MapToDomain
from uuid import UUID

class AccountRepo:
    @staticmethod
    def create_account(email: str, firstName: str, lastName: str, passwordHash: str, salt: str) -> AccountDomain:
        """Create a new account"""
        account = AccountDB(
            email=email,
            firstName=firstName,
            lastName=lastName,
            passwordHash=passwordHash,
            salt=salt,
            createdAt=datetime.now(),
            updatedAt=datetime.now()
        )
        db.session.add(account)
        db.session.commit()
        return MapToDomain.map_account(account)

    @staticmethod
    def get_account_by_email(email: str) -> AccountDomain:
        """Get account by email"""
        account = AccountDB.query.filter_by(email=email).first()
        if not account:
            return None
        return MapToDomain.map_account(account)

    @staticmethod
    def get_account_by_id(account_id: UUID) -> AccountDomain:
        """Get account by ID"""
        account = AccountDB.query.get(account_id.bytes)
        if not account:
            return None
        return MapToDomain.map_account(account)

    @staticmethod
    def update_lastLogin(account_id: UUID) -> None:
        """Update account's last login timestamp"""
        account = AccountDB.query.get(account_id.bytes)
        if account:
            account.lastLogin = datetime.now()
            account.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def update_password(account_id: UUID, passwordHash: str, salt: str) -> None:
        """Update account's password"""
        account = AccountDB.query.get(account_id.bytes)
        if account:
            account.passwordHash = passwordHash
            account.salt = salt
            account.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def set_mfaCode(account_id: UUID, mfaCode: str, expires_at: datetime) -> None:
        """Set MFA code for account"""
        print('account_id', account_id, type(account_id))
        account = AccountDB.query.get(account_id.bytes)
        if account:
            account.mfaCode = mfaCode
            account.mfaCodeExpires = expires_at
            db.session.commit()

    @staticmethod
    def enable_mfa(account_id: UUID) -> None:
        """Enable MFA for account"""
        account = AccountDB.query.get(account_id.bytes)
        if account:
            account.mfaEnabled = True
            account.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def verify_mfaCode(account_id: UUID, mfaCode: str) -> bool:
        """Verify MFA code"""
        account = AccountDB.query.get(account_id.bytes)
        if not account or not account.mfaCode:
            return False
        if account.mfaCodeExpires and datetime.now() > account.mfaCodeExpires:
            return False
        return account.mfaCode == mfaCode

    @staticmethod
    def account_exists(email: str) -> bool:
        """Check if account with email exists"""
        return AccountDB.query.filter_by(email=email).first() is not None

    @staticmethod
    def get_all_accounts():
        """Get all accounts (admin only)"""
        accounts = AccountDB.query.all()
        return [ MapToDomain.map_account(account) for account in accounts]

