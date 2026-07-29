"""Database operations for email auto-labeler."""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import DATABASE_FILE


class EmailDatabase:
    """Handles all database operations for email processing."""

    def __init__(
        self, conn: Optional[sqlite3.Connection] = None, database_file: str = DATABASE_FILE
    ):
        """Initialize database connection and create tables if needed.

        Args:
            conn: Optional SQLite connection object. If not provided, creates a new connection.
            database_file: Path to database file (used only if conn is not provided).
        """
        self.database_file = database_file
        if conn is not None:
            self.conn = conn
            self.owns_connection = False
        else:
            self.conn = sqlite3.connect(database_file)
            self.owns_connection = True
        self.cursor = self.conn.cursor()
        self.initialize_db()

    def initialize_db(self):
        """Initialize the SQLite database and create the necessary tables."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_emails (
                email_id TEXT PRIMARY KEY,
                processed_date DATETIME
            )
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS email_labels (
                email_id TEXT PRIMARY KEY,
                labels TEXT,
                category TEXT,
                last_updated DATETIME
            )
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS label_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT,
                old_labels TEXT,
                new_labels TEXT,
                timestamp DATETIME
            )
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                subject TEXT,
                sender TEXT,
                received_date DATETIME,
                content TEXT
            )
        """
        )
        try:
            self.cursor.execute("ALTER TABLE emails ADD COLUMN headers TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            self.cursor.execute("ALTER TABLE emails ADD COLUMN has_unsubscribe INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        self.conn.commit()
        logging.info("Database initialized successfully")

    def get_unprocessed_emails(self, limit: int = 100) -> List[tuple]:
        """Fetch unprocessed emails from the database."""
        self.cursor.execute(
            """
            SELECT e.id, e.subject, e.sender, e.received_date, e.content, e.headers, e.has_unsubscribe
            FROM emails e
            LEFT JOIN processed_emails p ON e.id = p.email_id
            WHERE p.email_id IS NULL
            ORDER BY e.received_date ASC
            LIMIT ?
        """,
            (limit,),
        )
        return self.cursor.fetchall()

    def update_email_labels(self, email_id: str, category: str, label_ids: List[str]):
        """Update email labels in the database and mark as processed."""
        current_time = datetime.now().isoformat()

        # Update or insert email labels
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO email_labels (email_id, labels, category, last_updated)
            VALUES (?, ?, ?, ?)
        """,
            (email_id, json.dumps(label_ids), category, current_time),
        )

        # Record history
        self.cursor.execute(
            """
            INSERT INTO label_history (email_id, old_labels, new_labels, timestamp)
            VALUES (?,
                    (SELECT labels FROM email_labels WHERE email_id = ?),
                    ?,
                    ?)
        """,
            (email_id, email_id, json.dumps(label_ids), current_time),
        )

        # Mark email as processed
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO processed_emails (email_id, processed_date)
            VALUES (?, ?)
        """,
            (email_id, current_time),
        )

        self.conn.commit()
        logging.info(f"Updated labels for email {email_id} with category {category}")

    def is_email_processed(self, email_id: str) -> bool:
        """Check if an email has already been processed."""
        self.cursor.execute(
            """
            SELECT email_id FROM processed_emails WHERE email_id = ?
        """,
            (email_id,),
        )
        return self.cursor.fetchone() is not None

    def get_email_labels(self, email_id: str) -> Optional[Tuple[str, List[str]]]:
        """Get the category and labels for a specific email."""
        self.cursor.execute(
            """
            SELECT category, labels FROM email_labels WHERE email_id = ?
        """,
            (email_id,),
        )
        result = self.cursor.fetchone()
        if result:
            category, labels_json = result
            return category, json.loads(labels_json)
        return None

    def save_email(
        self,
        email_id: str,
        subject: str,
        sender: str,
        received_date: str,
        content: str,
        headers: Dict = {},  # noqa: B006 — never mutated, json.dumps only reads it
        has_unsubscribe: bool = False,
    ):
        """Save email to the database."""
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO emails
                (id, subject, sender, received_date, content, headers, has_unsubscribe)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (email_id, subject, sender, received_date, content, json.dumps(headers),
             int(has_unsubscribe)),
        )
        self.conn.commit()

    def get_all_email_ids(self) -> List[str]:
        """Return all email_ids that have been labeled."""
        self.cursor.execute("SELECT email_id FROM email_labels")
        return [row[0] for row in self.cursor.fetchall()]

    def get_all_email_metadata(self) -> List[Tuple]:
        """Return id, subject, sender, headers, has_unsubscribe for all emails."""
        self.cursor.execute(
            "SELECT id, subject, sender, headers, has_unsubscribe FROM emails"
        )
        return self.cursor.fetchall()

    def close(self):
        """Close the database connection."""
        if self.owns_connection:
            self.conn.close()
            logging.info("Database connection closed")
        else:
            logging.debug("Skipping close - connection not owned by this instance")
