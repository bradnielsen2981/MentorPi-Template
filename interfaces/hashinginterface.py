import hashlib, hmac, secrets

#-------------Hashing----------------------------#
# Hashing is one-way: a hash can't be turned back into the password.
# To check a login, we hash the attempt the same way and compare the results.
#
# PBKDF2 runs SHA-256 many times over, so each guess is slow for an attacker.
# The random salt makes identical passwords produce different hashes.
# Stored format: "pbkdf2_sha256$<iterations>$<salt>$<hash>"

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 600_000

def _pbkdf2(password, salt, iterations):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations).hex()

#for hashing a password before storing it in the database
def hash_password(password):
    salt = secrets.token_hex(16)
    hashed = _pbkdf2(password, salt, ITERATIONS)
    return f"{ALGORITHM}${ITERATIONS}${salt}${hashed}"

#for checking a login attempt against the stored hash - returns True if correct
def check_password(hashed_password, user_password):
    try:
        if hashed_password.startswith(ALGORITHM + "$"):
            _, iterations, salt, stored = hashed_password.split("$")
            attempt = _pbkdf2(user_password, salt, int(iterations))
        else:
            # old format "hash:salt" (single SHA-256) - still accepted so existing accounts can log in
            stored, salt = hashed_password.split(":")
            attempt = hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()
    except (AttributeError, ValueError):
        return False  # missing or badly formatted hash
    # compare_digest takes the same time whether or not the hashes match, so timing gives nothing away
    return hmac.compare_digest(attempt, stored)
