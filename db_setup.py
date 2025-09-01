import sqlite3

conn = sqlite3.connect("users.db")  # This will create the file if it doesn't exist
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

conn.commit()
conn.close()
print("✅ Database initialized successfully.")
