from flask import request, session
from flask_restful import Resource
from services.accountService import AccountService

class AccountRegisterRouter(Resource):
    def post(self):
        """Register a new account"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = AccountService.register_account(
            email=data.get('email'),
            firstName=data.get('firstName'),
            lastName=data.get('lastName'),
            password=data.get('password')
        )

        status_code = 201 if result['success'] else 400
        return result, status_code

class AccountLoginRouter(Resource):
    def post(self):
        """Login account"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = AccountService.login(
            email=data.get('email'),
            password=data.get('password')
        )

        if result['success']:
            # Store account ID in session
            if 'account_id' in result or 'account' in result:
                account_id = result.get('account_id') or result['account'].id
                session['account_id'] = account_id
                session['email'] = data.get('email')
            status_code = 200
        else:
            status_code = 401

        return result, status_code

class AccountMFAVerifyRouter(Resource):
    def post(self):
        """Verify MFA code"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = AccountService.verify_mfa(
            account_id=data.get('account_id'),
            mfaCode=data.get('mfaCode')
        )

        if result['success']:
            # Store account ID in session
            session['account_id'] = data.get('account_id')
            session['email'] = result['account'].email
            status_code = 200
        else:
            status_code = 401

        return result, status_code

class AccountLogoutRouter(Resource):
    def post(self):
        """Logout account"""
        session.clear()
        return {'success': True, 'message': 'Logout successful'}, 200

class AccountMeRouter(Resource):
    def get(self):
        """Get current account info"""
        if 'account_id' not in session:
            # Return 200 so public pages can call /auth/me without treating
            # unauthenticated as an error in the frontend.
            return {'success': False, 'message': 'Not authenticated'}, 200

        from repository.accountRepo import AccountRepo
        account = AccountRepo.get_account_by_id(session['account_id'])

        if not account:
            return {'success': False, 'message': 'Account not found'}, 404

        return {'success': True, 'account': account.to_dict()}, 200

class AccountForgotPasswordRouter(Resource):
    def post(self):
        """Request password reset"""
        data = request.get_json()

        if not data or not data.get('email'):
            return {'success': False, 'message': 'Email is required'}, 400

        result = AccountService.request_password_reset(data.get('email'))
        return result, 200

class AccountResetPasswordRouter(Resource):
    def post(self):
        """Reset password with reset code"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = AccountService.reset_password(
            reset_code=data.get('token'),
            new_password=data.get('newPassword')
        )

        status_code = 200 if result['success'] else 400
        return result, status_code

class AccountEnableMFARouter(Resource):
    def post(self):
        """Enable MFA for account"""
        if 'account_id' not in session:
            return {'success': False, 'message': 'Not authenticated'}, 401

        result = AccountService.enable_mfa_for_account(session['account_id'])
        status_code = 200 if result['success'] else 400
        return result, status_code

