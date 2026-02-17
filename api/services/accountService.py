import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from repository.accountRepo import AccountRepo
from domain.account import Account
import os

PASSWORD_MIN_LENGTH = os.getenv('PASSWORD_MIN_LENGTH', 12)

# The encode() method encodes the string, using the specified encoding. If no encoding is specified, UTF-8 will be used.

class AccountService:
    # Email configuration (using free SMTP services)
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp-mail.outlook.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')

    @staticmethod
    def hash_password(password: str) -> tuple:
        """
        Hash a password with a salt
        Returns tuple of (passwordHash, salt)
        """
        salt = secrets.token_hex(16)
        passwordHash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return passwordHash.hex(), salt

    @staticmethod
    def verify_password(password: str, passwordHash: str, salt: str) -> bool:
        """Verify a password against a hash"""
        passwordHash_attempt = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return passwordHash_attempt.hex() == passwordHash

    @staticmethod
    def register_account(email: str, firstName: str, lastName: str, password: str) -> dict:
        """Register a new account"""
        # Validate input
        if not email or not firstName or not lastName or not password:
            return {'success': False, 'message': 'All fields are required'}

        if AccountRepo.account_exists(email):
            return {'success': False, 'message': 'Account with this email already exists'}

        if len(password) < PASSWORD_MIN_LENGTH:
            return {'success': False, 'message': 'Password must be at least 8 characters long'}

        # Hash password
        passwordHash, salt = AccountService.hash_password(password)

        # Create account
        account = AccountRepo.create_account(email, firstName, lastName, passwordHash, salt)
        return {
            'success': True,
            'message': 'Account registered successfully',
            'account': account
        }

    @staticmethod
    def login(email: str, password: str) -> dict:
        """Login account"""
        account = AccountRepo.get_account_by_email(email)
        if not account:
            return {'success': False, 'message': 'Invalid email or password'}

        if not AccountService.verify_password(password, account.passwordHash, account.salt):
            return {'success': False, 'message': 'Invalid email or password'}

        # Update last login
        AccountRepo.update_lastLogin(account.id)

        # Check if MFA is enabled
        if account.mfaEnabled:
            # Generate and send MFA code
            mfaCode = AccountService.generate_mfaCode()
            expires_at = datetime.now() + timedelta(minutes=10)
            AccountRepo.set_mfaCode(account.id, mfaCode, expires_at)

            # Try to send email
            if AccountService.SENDER_EMAIL and AccountService.SENDER_PASSWORD:
                AccountService.send_mfa_email(email, mfaCode)
                return {
                    'success': True,
                    'message': 'MFA code sent to email',
                    'requires_mfa': True,
                    'account_id': account.id,
                }
            else:
                return {
                    'success': True,
                    'message': 'MFA enabled but email not configured',
                    'requires_mfa': True,
                    'account_id': account.id,
                    'mfaCode': mfaCode  # Return code for testing (remove in production)
                }

        return {
            'success': True,
            'message': 'Login successful',
            'account': account,
            'requires_mfa': False,
        }

    @staticmethod
    def verify_mfa(account_id: int, mfaCode: str) -> dict:
        """Verify MFA code"""
        if not AccountRepo.verify_mfaCode(account_id, mfaCode):
            return {'success': False, 'message': 'Invalid or expired MFA code'}

        account = AccountRepo.get_account_by_id(account_id)
        if not account:
            return {'success': False, 'message': 'Account not found'}

        # Clear MFA code
        AccountRepo.set_mfaCode(account_id, None, None)

        return {
            'success': True,
            'message': 'MFA verification successful',
            'account': account,
        }

    @staticmethod
    def generate_mfaCode() -> str:
        """Generate a 6-digit MFA code"""
        return ''.join(secrets.choice('0123456789') for _ in range(6))

    @staticmethod
    def send_mfa_email(email: str, mfaCode: str) -> bool:
        """Send MFA code via email (DISABLED)"""
        if not AccountService.SENDER_EMAIL or not AccountService.SENDER_PASSWORD:
            return False

        try:
            message = MIMEMultipart()
            message['From'] = AccountService.SENDER_EMAIL
            message['To'] = email
            message['Subject'] = 'Your MFA Code'

            body = f"""
            Your MFA code is: {mfaCode}

            This code will expire in 10 minutes.

            If you did not request this code, please ignore this email.
            """

            message.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(AccountService.SMTP_SERVER, AccountService.SMTP_PORT) as server:
                server.starttls()
                server.login(AccountService.SENDER_EMAIL, AccountService.SENDER_PASSWORD)
                server.send_message(message)

            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    @staticmethod
    def request_password_reset(email: str) -> dict:
        """Request password reset"""
        account = AccountRepo.get_account_by_email(email)
        if not account:
            # Don't reveal if email exists for security
            return {'success': True, 'message': 'If email exists, reset link has been sent'}

        # Generate reset code
        reset_code = AccountService.generate_mfaCode()
        expires_at = datetime.now() + timedelta(hours=1)

        # In a real app, you'd store this in a separate table
        # For now, we'll use MFA code field temporarily
        AccountRepo.set_mfaCode(account.id, reset_code, expires_at)

        # Send reset email
        if AccountService.send_password_reset_email(email, reset_code):
            return {'success': True, 'message': 'Password reset link has been sent to email'}
        else:
            return {'success': True, 'message': 'Email service unavailable, but reset code generated', 'reset_code': reset_code}

    @staticmethod
    def reset_password(reset_code: str, new_password: str) -> dict:
        """Reset password with reset code"""
        if len(new_password) < 8:
            return {'success': False, 'message': 'Password must be at least 8 characters long'}

        # Find account with this reset code
        # In a real app, query the password reset table
        # For now, we search through accounts
        accounts = AccountRepo.get_all_accounts()
        account_found = None

        for account in accounts:
            if account.mfaCode == reset_code:
                account_found = account
                break

        if not account_found:
            return {'success': False, 'message': 'Invalid or expired reset code'}

        # Hash new password
        passwordHash, salt = AccountService.hash_password(new_password)
        AccountRepo.update_password(account_found.id, passwordHash, salt)

        # Clear reset code
        AccountRepo.set_mfaCode(account_found.id, None, None)

        return {'success': True, 'message': 'Password reset successful'}

    @staticmethod
    def send_password_reset_email(email: str, reset_code: str) -> bool:
        """Send password reset email"""
        if not AccountService.SENDER_EMAIL or not AccountService.SENDER_PASSWORD:
            return False

        try:
            message = MIMEMultipart()
            message['From'] = AccountService.SENDER_EMAIL
            message['To'] = email
            message['Subject'] = 'Password Reset Request'

            # In a real app, include a link with the reset code
            body = f"""
            You requested a password reset.

            Reset code: {reset_code}

            This code will expire in 1 hour.

            If you did not request this, please ignore this email.
            """

            message.attach(MIMEText(body, 'plain'))

            # MFA TEMP DISABLED
            print(f'Password reset: {reset_code} - {email}')
            # with smtplib.SMTP(AccountService.SMTP_SERVER, AccountService.SMTP_PORT) as server:
            #     server.starttls()
            #     server.login(AccountService.SENDER_EMAIL, AccountService.SENDER_PASSWORD)
            #     server.send_message(message)

            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    @staticmethod
    def enable_mfa_for_account(account_id: int) -> dict:
        """Enable MFA for account"""
        account = AccountRepo.get_account_by_id(account_id)
        if not account:
            return {'success': False, 'message': 'Account not found'}

        AccountRepo.enable_mfa(account_id)
        return {'success': True, 'message': 'MFA enabled successfully'}

