from repository.db import db
from repository.models import Account as AccountDB
from domain.account import Account as AccountDomain
from datetime import datetime
from repository.MapToDomain import MapToDomain

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
    def get_account_by_id(account_id: int) -> AccountDomain:
        """Get account by ID"""
        account = AccountDB.query.get(account_id)
        if not account:
            return None
        return MapToDomain.map_account(account)

    @staticmethod
    def update_lastLogin(account_id: int) -> None:
        """Update account's last login timestamp"""
        account = AccountDB.query.get(account_id)
        if account:
            account.lastLogin = datetime.now()
            account.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def update_password(account_id: int, passwordHash: str, salt: str) -> None:
        """Update account's password"""
        account = AccountDB.query.get(account_id)
        if account:
            account.passwordHash = passwordHash
            account.salt = salt
            account.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def set_mfaCode(account_id: int, mfaCode: str, expires_at: datetime) -> None:
        """Set MFA code for account"""
        account = AccountDB.query.get(account_id)
        if account:
            account.mfaCode = mfaCode
            account.mfaCodeExpires = expires_at
            db.session.commit()

    @staticmethod
    def enable_mfa(account_id: int) -> None:
        """Enable MFA for account"""
        account = AccountDB.query.get(account_id)
        if account:
            account.mfaEnabled = True
            account.updatedAt = datetime.now()
            db.session.commit()

    @staticmethod
    def verify_mfaCode(account_id: int, mfaCode: str) -> bool:
        """Verify MFA code"""
        account = AccountDB.query.get(account_id)
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

