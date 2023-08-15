import psycopg2

# Connect to an existing database
conn = psycopg2.connect("dbname=hacker_news user=matt password=postgres")

# Open a cursor to perform database operations
cur = conn.cursor()
# cur.execute("DROP TABLE IF EXISTS hacker_news")

# Execute a command: this creates a new table
cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY,
        by TEXT,
        title TEXT,
        time TIMESTAMP,
        type TEXT,
        url TEXT,
        score INTEGER,
        descendants INTEGER,
        kids INTEGER[]
           );""")