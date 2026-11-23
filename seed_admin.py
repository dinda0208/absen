"""Membuat akun admin. Pakai:  python seed_admin.py [username] [password]"""
import sys
from werkzeug.security import generate_password_hash
import db

username = sys.argv[1] if len(sys.argv) > 1 else "admin"
password = sys.argv[2] if len(sys.argv) > 2 else "admin123"

ada = db.query("SELECT id FROM users WHERE username=%s", (username,), one=True)
if ada:
    db.execute("UPDATE users SET password_hash=%s WHERE id=%s", (generate_password_hash(password), ada["id"]))
    print(f"Password admin '{username}' diperbarui.")
else:
    db.execute(
        "INSERT INTO users (kode_karyawan,nama_lengkap,jabatan,username,password_hash,role) "
        "VALUES ('ADM001','Administrator','Admin',%s,%s,'admin')",
        (username, generate_password_hash(password)),
    )
    print(f"Admin dibuat. Username: {username} | Password: {password}")
