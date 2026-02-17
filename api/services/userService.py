import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from repository.userRepo import UserRepo
from domain.user import User
import os

PASSWORD_MIN_LENGTH = os.getenv('PASSWORD_MIN_LENGTH', 12)

# The encode() method encodes the string, using the specified encoding. If no encoding is specified, UTF-8 will be used.

class UserService:
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
    def register_user(email: str, firstName: str, lastName: str, password: str) -> dict:
        """Register a new user"""
        # Validate input
        if not email or not firstName or not lastName or not password:
            return {'success': False, 'message': 'All fields are required'}

        if UserRepo.user_exists(email):
            return {'success': False, 'message': 'User with this email already exists'}

        if len(password) < PASSWORD_MIN_LENGTH:
            return {'success': False, 'message': 'Password must be at least 8 characters long'}

        # Hash password
        passwordHash, salt = UserService.hash_password(password)

        # Create user
        user = UserRepo.create_user(email, firstName, lastName, passwordHash, salt)
        return {
            'success': True,
            'message': 'User registered successfully',
            'user': user
        }

    @staticmethod
    def login(email: str, password: str) -> dict:
        """Login user"""
        user = UserRepo.get_user_by_email(email)
        if not user:
            return {'success': False, 'message': 'Invalid email or password'}

        if not UserService.verify_password(password, user.passwordHash, user.salt):
            return {'success': False, 'message': 'Invalid email or password'}

        # Update last login
        UserRepo.update_lastLogin(user.id)

        # Check if MFA is enabled
        if user.mfaEnabled:
            # Generate and send MFA code
            mfaCode = UserService.generate_mfaCode()
            expires_at = datetime.now() + timedelta(minutes=10)
            UserRepo.set_mfaCode(user.id, mfaCode, expires_at)

            # Try to send email
            if UserService.SENDER_EMAIL and UserService.SENDER_PASSWORD:
                UserService.send_mfa_email(email, mfaCode)
                return {
                    'success': True,
                    'message': 'MFA code sent to email',
                    'requires_mfa': True,
                    'user_id': user.id,
                }
            else:
                return {
                    'success': True,
                    'message': 'MFA enabled but email not configured',
                    'requires_mfa': True,
                    'user_id': user.id,
                    'mfaCode': mfaCode  # Return code for testing (remove in production)
                }

        return {
            'success': True,
            'message': 'Login successful',
            'user': user,
            'requires_mfa': False,
        }

    @staticmethod
    def verify_mfa(user_id: int, mfaCode: str) -> dict:
        """Verify MFA code"""
        if not UserRepo.verify_mfaCode(user_id, mfaCode):
            return {'success': False, 'message': 'Invalid or expired MFA code'}

        user = UserRepo.get_user_by_id(user_id)
        if not user:
            return {'success': False, 'message': 'User not found'}

        # Clear MFA code
        UserRepo.set_mfaCode(user_id, None, None)

        return {
            'success': True,
            'message': 'MFA verification successful',
            'user': user,
        }

    @staticmethod
    def generate_mfaCode() -> str:
        """Generate a 6-digit MFA code"""
        return ''.join(secrets.choice('0123456789') for _ in range(6))

    @staticmethod
    def send_mfa_email(email: str, mfaCode: str) -> bool:
        """Send MFA code via email (DISABLED)"""
        if not UserService.SENDER_EMAIL or not UserService.SENDER_PASSWORD:
            return False

        try:
            message = MIMEMultipart()
            message['From'] = UserService.SENDER_EMAIL
            message['To'] = email
            message['Subject'] = 'Your MFA Code'

            body = f"""
            Your MFA code is: {mfaCode}

            This code will expire in 10 minutes.

            If you did not request this code, please ignore this email.
            """

            message.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(UserService.SMTP_SERVER, UserService.SMTP_PORT) as server:
                server.starttls()
                server.login(UserService.SENDER_EMAIL, UserService.SENDER_PASSWORD)
                server.send_message(message)

            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    @staticmethod
    def request_password_reset(email: str) -> dict:
        """Request password reset"""
        user = UserRepo.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists for security
            return {'success': True, 'message': 'If email exists, reset link has been sent'}

        # Generate reset code
        reset_code = UserService.generate_mfaCode()
        expires_at = datetime.now() + timedelta(hours=1)

        # In a real app, you'd store this in a separate table
        # For now, we'll use MFA code field temporarily
        UserRepo.set_mfaCode(user.id, reset_code, expires_at)

        # Send reset email
        if UserService.send_password_reset_email(email, reset_code):
            return {'success': True, 'message': 'Password reset link has been sent to email'}
        else:
            return {'success': True, 'message': 'Email service unavailable, but reset code generated', 'reset_code': reset_code}

    @staticmethod
    def reset_password(reset_code: str, new_password: str) -> dict:
        """Reset password with reset code"""
        if len(new_password) < 8:
            return {'success': False, 'message': 'Password must be at least 8 characters long'}

        # Find user with this reset code
        # In a real app, query the password reset table
        # For now, we search through users
        users = UserRepo.get_all_users()
        user_found = None

        for user in users:
            if user.mfaCode == reset_code:
                user_found = user
                break

        if not user_found:
            return {'success': False, 'message': 'Invalid or expired reset code'}

        # Hash new password
        passwordHash, salt = UserService.hash_password(new_password)
        UserRepo.update_password(user_found.id, passwordHash, salt)

        # Clear reset code
        UserRepo.set_mfaCode(user_found.id, None, None)

        return {'success': True, 'message': 'Password reset successful'}

    @staticmethod
    def send_password_reset_email(email: str, reset_code: str) -> bool:
        """Send password reset email"""
        if not UserService.SENDER_EMAIL or not UserService.SENDER_PASSWORD:
            return False

        try:
            message = MIMEMultipart()
            message['From'] = UserService.SENDER_EMAIL
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
            # with smtplib.SMTP(UserService.SMTP_SERVER, UserService.SMTP_PORT) as server:
            #     server.starttls()
            #     server.login(UserService.SENDER_EMAIL, UserService.SENDER_PASSWORD)
            #     server.send_message(message)

            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    @staticmethod
    def enable_mfa_for_user(user_id: int) -> dict:
        """Enable MFA for user"""
        user = UserRepo.get_user_by_id(user_id)
        if not user:
            return {'success': False, 'message': 'User not found'}

        UserRepo.enable_mfa(user_id)
        return {'success': True, 'message': 'MFA enabled successfully'}

