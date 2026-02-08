from flask import request, session
from flask_restful import Resource
from services.userService import UserService

class UserRegisterRouter(Resource):
    def post(self):
        """Register a new user"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = UserService.register_user(
            email=data.get('email'),
            firstName=data.get('firstName'),
            lastName=data.get('lastName'),
            password=data.get('password')
        )

        status_code = 201 if result['success'] else 400
        return result, status_code

class UserLoginRouter(Resource):
    def post(self):
        """Login user"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = UserService.login(
            email=data.get('email'),
            password=data.get('password')
        )

        if result['success']:
            # Store user ID in session
            if 'user_id' in result or 'user' in result:
                user_id = result.get('user_id') or result['user'].get('id')
                session['user_id'] = user_id
                session['email'] = data.get('email')
            status_code = 200
        else:
            status_code = 401

        return result, status_code

class UserMFAVerifyRouter(Resource):
    def post(self):
        """Verify MFA code"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = UserService.verify_mfa(
            user_id=data.get('user_id'),
            mfaCode=data.get('mfaCode')
        )

        if result['success']:
            # Store user ID in session
            session['user_id'] = data.get('user_id')
            session['email'] = result['user'].get('email')
            status_code = 200
        else:
            status_code = 401

        return result, status_code

class UserLogoutRouter(Resource):
    def post(self):
        """Logout user"""
        session.clear()
        return {'success': True, 'message': 'Logout successful'}, 200

class UserMeRouter(Resource):
    def get(self):
        """Get current user info"""
        if 'user_id' not in session:
            # Return 200 so public pages can call /auth/me without treating
            # unauthenticated as an error in the frontend.
            return {'success': False, 'message': 'Not authenticated'}, 200

        from repository.userRepo import UserRepo
        user = UserRepo.get_user_by_id(session['user_id'])

        if not user:
            return {'success': False, 'message': 'User not found'}, 404

        return {'success': True, 'user': user.to_dict()}, 200

class UserForgotPasswordRouter(Resource):
    def post(self):
        """Request password reset"""
        data = request.get_json()

        if not data or not data.get('email'):
            return {'success': False, 'message': 'Email is required'}, 400

        result = UserService.request_password_reset(data.get('email'))
        return result, 200

class UserResetPasswordRouter(Resource):
    def post(self):
        """Reset password with reset code"""
        data = request.get_json()

        if not data:
            return {'success': False, 'message': 'Invalid request'}, 400

        result = UserService.reset_password(
            reset_code=data.get('token'),
            new_password=data.get('newPassword')
        )

        status_code = 200 if result['success'] else 400
        return result, status_code

class UserEnableMFARouter(Resource):
    def post(self):
        """Enable MFA for user"""
        if 'user_id' not in session:
            return {'success': False, 'message': 'Not authenticated'}, 401

        result = UserService.enable_mfa_for_user(session['user_id'])
        status_code = 200 if result['success'] else 400
        return result, status_code

