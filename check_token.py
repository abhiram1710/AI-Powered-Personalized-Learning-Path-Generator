import auth_db
import time

with auth_db.connect() as db:
    row = db.execute(
        "SELECT * FROM password_reset_tokens ORDER BY id DESC LIMIT 1"
    ).fetchone()

print(dict(row))
print("CURRENT:", time.time())
print("EXPIRES:", row["expires_at"])
print("REMAINING:", row["expires_at"] - time.time())

