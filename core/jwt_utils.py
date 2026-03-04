import jwt
import datetime

SECRET_KEY = "contexguard_secure_secret_key" # In production, load from env
ALGORITHM = "HS256"

def generate_token(user_id: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        "iss": "contexguard_pdp"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def validate_token(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded_token
    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")
