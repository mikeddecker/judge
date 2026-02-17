# Authentication & Login System Documentation

## Overview

A complete account authentication and session management system has been implemented with:
- Account registration and login
- Password hashing with salt
- Multi-Factor Authentication (MFA) via email
- Forgot password functionality
- Secure session management
- Account profile management

## Features

### 1. Account Registration
- **Endpoint**: `POST /auth/register`
- **Fields**: email, firstName, lastName, password
- **Validation**: Password must be at least 8 characters
- **Response**: Account object with ID and creation timestamp

```bash
curl -X POST http://localhost:5555/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "account@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "password": "SecurePass123"
  }'
```

### 2. Account Login
- **Endpoint**: `POST /auth/login`
- **Fields**: email, password
- **Returns**: Account object, session token, or MFA requirement
- **MFA Flow**: If enabled, returns `requires_mfa: true` with account_id

```bash
curl -X POST http://localhost:5555/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "account@example.com",
    "password": "SecurePass123"
  }'
```

### 3. Multi-Factor Authentication (MFA) - DISABLED
#### Enabling MFA
- **Endpoint**: `POST /auth/enable-mfa`
- **Requires**: Active session (authentication)

#### MFA Verification
- **Endpoint**: `POST /auth/mfa/verify`
- **Fields**: account_id, mfaCode
- **MFA Code**: 6-digit code sent via email
- **Expiration**: 10 minutes

```bash
curl -X POST http://localhost:5555/auth/mfa/verify \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": 1,
    "mfaCode": "123456"
  }'
```

### 4. Forgot Password
- **Endpoint**: `POST /auth/forgot-password`
- **Fields**: email
- **Process**:
  1. Account requests password reset with email
  2. System generates reset code (valid for 1 hour)
  3. Reset code sent via email (if email configured)
  4. Account receives code and new password endpoint

```bash
curl -X POST http://localhost:5555/auth/forgot-password \
  -H "Content-Type: application/json" \
  -d '{"email": "account@example.com"}'
```

### 5. Reset Password
- **Endpoint**: `POST /auth/reset-password`
- **Fields**: reset_code, new_password
- **Validation**: Password must be at least 8 characters

```bash
curl -X POST http://localhost:5555/auth/reset-password \
  -H "Content-Type: application/json" \
  -d '{
    "reset_code": "token_from_email",
    "new_password": "NewSecurePass123"
  }'
```

### 6. Session Management
- **Endpoint**: `GET /auth/me`
- **Purpose**: Get current authenticated account info
- **Requires**: Valid session

```bash
curl http://localhost:5555/auth/me
```

### 7. Logout
- **Endpoint**: `POST /auth/logout`
- **Purpose**: Clear session and logout account

```bash
curl -X POST http://localhost:5555/auth/logout
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Flask Session Configuration
SECRET_KEY=your-secret-key-change-in-production
SESSION_COOKIE_SECURE=false  # Set to true in production
FLASK_ENV=development

# Email Configuration (for MFA and Password Reset)
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
SENDER_EMAIL=your-email@outlook.com
SENDER_PASSWORD=your-app-password

# Authentication
REQUIRE_MFA=false
```

### Session Configuration (in api/config.py)

Session and authentication settings are configured in the `Config` and `TestConfig` classes:

- `PERMANENT_SESSION_LIFETIME = 86400 * 7` (7 days)
- `SESSION_COOKIE_SECURE = True` (production) / `False` (development)
- `SESSION_COOKIE_HTTPONLY = True` (prevents JS access)
- `SESSION_COOKIE_SAMESITE = 'Lax'` (CSRF protection)

## Frontend Components

### Login View
- **Path**: `src/views/LoginView.vue`
- **Features**:
  - Email/password login form
  - MFA code input (if needed)
  - Error handling
  - Links to register and forgot password

### Register View
- **Path**: `src/views/RegisterView.vue`
- **Features**:
  - Registration form with validation
  - Password confirmation
  - Minimum password length validation

### Forgot Password View
- **Path**: `src/views/ForgotPasswordView.vue`
- **Features**:
  - Two-step process (request reset → reset password)
  - Email input
  - Reset code verification
  - New password entry

### Authentication Store
- **Path**: `src/stores/authStore.js`
- **Methods**:
  - `setAccount(accountData)` - Set authenticated account
  - `clearAccount()` - Clear account data
  - `logout()` - Logout account
  - `enableMFA()` - Enable MFA for account
  - `initializeAuth()` - Load account from session on startup

## Security Features

1. **Password Hashing**: Uses PBKDF2 with SHA256
2. **Salting**: Random salt generated for each account
3. **Session Management**: Secure HTTP-only cookies
4. **MFA**: 6-digit codes with 10-minute expiration
5. **HTTPS**: Recommended for production
6. **CORS**: Configured to prevent unauthorized access

## Email Setup (Outlook Example) - DISABLED

To use MFA and password reset with Outlook/Hotmail:

1. Enable 2-Factor Authentication on your Gmail account
2. Generate an App Password (not your regular password):
   - Go to https://myaccount.google.com/apppasswords
   - Select "Mail" and "Windows Computer"
   - Generate and copy the password

3. Set in `.env`:
   ```env
   SENDER_EMAIL=your-email@gmail.com
   SENDER_PASSWORD=your-app-password-32-chars
   ```

## API Response Examples

### Successful Login (No MFA)
```json
{
  "success": true,
  "message": "Login successful",
  "requires_mfa": false,
  "account": {
    "id": 1,
    "email": "account@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "lastLogin": "2026-02-07T10:30:00",
    "createdAt": "2026-02-07T09:00:00",
    "updatedAt": "2026-02-07T10:30:00",
    "mfaEnabled": false
  }
}
```

### Login with MFA Required
```json
{
  "success": true,
  "message": "MFA code sent to email",
  "requires_mfa": true,
  "account_id": 1
}
```

### MFA Verification Error
```json
{
  "success": false,
  "message": "Invalid or expired MFA code"
}
```

## Troubleshooting

### MFA Email Not Received
1. Check `.env` file for correct SMTP settings
2. Verify email account supports SMTP
3. Check firewall/network blocking port 587
3. Create and use App Password from your email provider's security settings

### Session Not Persisting
1. Ensure `SECRET_KEY` is set in `.env`
2. Check browser cookies are enabled
3. Verify `SESSION_COOKIE_HTTPONLY` is not breaking your setup
4. For development, set `SESSION_COOKIE_SECURE=false`

### Password Reset Code Expired
1. Default expiration is 1 hour - account must reset within this time
2. Can be changed in `config.py` `PASSWORD_RESET_EXPIRATION`
3. Account can request a new code by clicking "Forgot Password" again

## ✅ Deployment Checklist

- [x] `SECRET_KEY` changed to strong random string
- [ ] Email configured with real credentials <-- Microsoft disabled SMTP app password, temp disabled MFA
- [x] `FLASK_ENV=production` set
- [x] `SESSION_COOKIE_SECURE=true` (HTTPS only)
- [x] Database backed up
- [ ] Rate limiting configured (recommended)
- [ ] Monitoring/logging enabled
- [ ] MFA tested end-to-end
- [ ] Password reset tested
- [ ] Load testing completed ? login works

## Future Enhancements

- [ ] Refresh token for longer-lived sessions
- [ ] Social login (Google, GitHub, etc.)
- [ ] TOTP-based MFA (Authenticator app)
- [ ] Password strength meter
- [ ] Login history and activity tracking
- [ ] IP-based device recognition
- [ ] Two-factor recovery codes
- [ ] Account lockout after failed attempts
- [ ] CORS whitelist configured -> prod

### For Security Audits
1. [ ] Review **AUTHENTICATION_COMPLETE.md** security section
2. [ ] Check **api/services/accountService.py** password hashing
3. [ ] Review **api/config.py** session configuration
4. [ ] Check **web/src/router/index.js** route guards
5. [ ] Review **COMPLETION_REPORT.md** security features & checklist
